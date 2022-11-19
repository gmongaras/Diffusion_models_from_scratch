import torch
from torch import nn
from .U_Net import U_Net
from ..helpers.image_rescale import reduce_image, unreduce_image
from ..blocks.PositionalEncoding import PositionalEncoding
import os
import json
import threading






class diff_model(nn.Module):
    # inCh - Number of input channels in the input batch
    # embCh - Number of channels to embed the batch to
    # chMult - Multiplier to scale the number of channels by
    #          for each up/down sampling block
    # num_heads - Number of heads in each multi-head non-local block
    # num_res_blocks - Number of residual blocks on the up/down path
    # T - Max number of diffusion steps
    # beta_sched - Scheduler for the beta noise term (linear or cosine)
    # useDeep - True to use deep residual blocks, False to use not deep residual blocks
    # t_dim - Embedding dimenion for the timesteps
    # device - Device to put the model on (gpu or cpu)
    def __init__(self, inCh, embCh, chMult, num_heads, num_res_blocks,
                 T, beta_sched, t_dim, device, useDeep=False):
        super(diff_model, self).__init__()
        
        self.beta_sched = beta_sched
        self.inCh = inCh
        
        # Important default parameters
        self.defaults = {
            "inCh": inCh,
            "embCh": embCh,
            "chMult": chMult,
            "num_heads": num_heads,
            "num_res_blocks": num_res_blocks,
            "T": T,
            "beta_sched": beta_sched,
            "t_dim": t_dim,
            "useDeep": useDeep
        }
        
        # Convert the device to a torch device
        if device.lower() == "gpu":
            if torch.cuda.is_available():
                dev = device.lower()
                device = torch.device('cuda:0')
            else:
                dev = "cpu"
                print("GPU not available, defaulting to CPU. Please ignore this message if you do not wish to use a GPU\n")
                device = torch.device('cpu')
        else:
            dev = device.lower()
            device = torch.device('cpu')
        self.device = device
        self.dev = dev
        
        # Convert T to a tensor
        self.T = torch.tensor(T, device=device)
        
        # U_net model
        self.unet = U_Net(inCh, inCh*2, embCh, chMult, t_dim, num_heads, num_res_blocks, useDeep).to(device)
        
        # What scheduler should be used to add noise
        # to the data?
        if self.beta_sched == "cosine":
            def f(t):
                s = 0.008
                return torch.cos(((t/T + s)/(1+s)) * (torch.pi/2))**2 /\
                    torch.cos(torch.tensor((s/(1+s)) * (torch.pi/2)))**2
            self.beta_sched_funct = f
        else: # Linear
            self.beta_sched_funct = torch.linspace(1e-4, 0.02, T)
            
        # Used to embed the values of t so the model can use it
        self.t_emb = PositionalEncoding(t_dim).to(device)

        # Output convolutions for the mean and variance
        self.out_mean = nn.Conv2d(inCh, inCh, 3, padding=1)
        self.out_var = nn.Conv2d(inCh, inCh, 3, padding=1)
            
            
            
    # Used to get the value of beta, a and a_bar from the schedulers
    # Inputs:
    #   t - Batch of t values of shape (N) 
    #       Note: t values can be in the range [0, T-1]
    # Outputs:
    #   Batch of beta and a values:
    #     beta_t
    #     a_t
    #     a_bar_t
    def get_scheduler_info(self, t):
        # Gradients don't matter here
        with torch.no_grad():
            # t value assertion
            assert torch.all(t <= self.T-1) and torch.all(t >= -1), "The value of t can be in the range [-1, T-1]"
            
            # Values depend on the scheduler
            if self.beta_sched == "cosine":
                # Beta_t, a_t, and a_bar_t
                # using the cosine scheduler
                a_bar_t = self.beta_sched_funct(t)
                a_bar_t1 = torch.where(t > 0, self.beta_sched_funct(t-1), a_bar_t)
                beta_t = 1-(a_bar_t/(a_bar_t1))
                beta_t = torch.clamp(beta_t, 0, 0.999)
                a_t = 1-beta_t
            else:
                # Beta_t, a_t, and a_bar_t
                # using the linear scheduler
                beta_t = self.beta_sched_funct.repeat(t.shape[0])[t]
                a_t = 1-beta_t
                a_bar_t = torch.zeros(t.shape[0])
                for b in range(0, t.shape[0]):
                    a_bar_t[b] = torch.prod(1-self.beta_sched_funct[:t[b]+1])
                
            return beta_t.to(self.device), a_t.to(self.device), a_bar_t.to(self.device)
    
    
    # Unsqueezing n times along the given dim.
    # Note: dim can be 0 or -1
    def unsqueeze(self, X, dim, n):
        if dim == 0:
            return X.reshape(n*(1,) + X.shape)
        else:
            return X.reshape(X.shape + (1,)*n)
    
        
    # Used to noise a batch of images by t timesteps
    # Inputs:
    #   X - Batch of images of shape (N, C, L, W)
    #   t - Batch of t values of shape (N)
    # Outputs:
    #   Batch of noised images of shape (N, C, L, W)
    #   Batch of noised images of shape (N, C, L, W)
    #   Noise added to the images of shape (N, C, L, W)
    def noise_batch(self, X, t):
        # Make sure t isn't too large
        t = torch.min(t, self.T)
        
        # Sample gaussian noise
        epsilon = torch.randn_like(X, device=self.device)
        
        # The value of a_bar_t at timestep t depending on the scheduler
        a_bar_t = self.unsqueeze(self.get_scheduler_info(t)[2], -1, 3)
        t = self.unsqueeze(t, -1, 3)
        
        # Noise the images
        return torch.sqrt(a_bar_t)*X + torch.sqrt(1-a_bar_t)*epsilon, epsilon
    
    
    
    # Get the noise for a batch of images
    # Inputs:
    #   noise_shape - Shape of desired tensor of noise
    #   t - Batch of t values of shape (N)
    # Outputs:
    #   epsilon - Batch of noised images of the given shape
    def sample_noise(self, noise_shape, t):
        # Make sure t isn't too large
        t = torch.min(t, self.T)
        
        # Sample gaussian noise
        epsilon = torch.randn(noise_shape, device=self.device)
        
        return epsilon
    
    
    # Used to convert a batch of noise predictions to
    # a batch of mean predictions
    # Inputs:
    #   epsilon - The epsilon value for the mean of shape (N, C, L, W)
    #   x_t - The image to unoise of shape (N, C, L, W)
    #   t - A batch of t values for the beta schedulers of shape (N)
    # Outputs:
    #   A tensor of shape (N, C, L, W) representing the mean of the
    #     unnoised image x_t-1
    def noise_to_mean(self, epsilon, x_t, t):
        # Note: Function from the following:
        # https://github.com/hojonathanho/diffusion/issues/5


        # Get the beta and a values for the batch of t values
        beta_t, a_t, a_bar_t = self.get_scheduler_info(t)

        # Get the previous a bar values for the batch of t-1 values
        a_bar_t1 = torch.where(t == 0, 0, self.get_scheduler_info(t-1)[2])

        # Make sure everything is in the correct shape
        beta_t = self.unsqueeze(beta_t, -1, 3)
        a_t = self.unsqueeze(a_t, -1, 3)
        a_bar_t = self.unsqueeze(a_bar_t, -1, 3)
        a_bar_t1 = self.unsqueeze(a_bar_t1, -1, 3)
        if len(t.shape) == 1:
            t = self.unsqueeze(t, -1, 3)


        # Calculate the mean and return it
        mean = torch.where(t == 0,
            # When t is 0, normal without correction
            (1/torch.sqrt(a_t))*(x_t - ((1-a_t)/torch.sqrt(1-a_bar_t))*epsilon),

            # When t is not 0, special with correction
            (torch.sqrt(a_bar_t1)*beta_t)/(1-a_bar_t) * \
                torch.clamp( (1/torch.sqrt(a_bar_t))*x_t - torch.sqrt((1-a_bar_t)/a_bar_t)*epsilon, -1, 1 ) + \
                (((1-a_bar_t1)*torch.sqrt(a_t))/(1-a_bar_t))*x_t
        )
        return mean
        #return (1/torch.sqrt(a_t))*(x_t - ((1-a_t)/torch.sqrt(1-a_bar_t))*epsilon)
        return (torch.sqrt(a_bar_t1)*beta_t)/(1-a_bar_t) * \
            torch.clamp( (1/torch.sqrt(a_bar_t))*x_t - torch.sqrt((1-a_bar_t)/a_bar_t)*epsilon, -1, 1 ) + \
            (((1-a_bar_t1)*torch.sqrt(a_t))/(1-a_bar_t))*x_t
    
    
    
    # Used to convert a batch of predicted v values to
    # a batch of variance predictions
    def vs_to_variance(self, v, t):
        # Get the beta values for this batch of ts
        beta_t, _, a_bar_t = self.get_scheduler_info(t)
        
        # Beta values for the previous value of t
        _, _, a_bar_t1 = self.get_scheduler_info(t-1)
        
        # Get the beta tilde value
        beta_tilde_t = ((1-a_bar_t1)/(1-a_bar_t))*beta_t
        
        beta_t = self.unsqueeze(beta_t, -1, 3)
        beta_tilde_t = self.unsqueeze(beta_tilde_t, -1, 3)

        """
        Note: The authors claim that v stayed in the range of values
        it should without any type of restraint. I found that this was
        the case at later stages of t, but at early stages of t (from about t = 20 to t = 0),
        the value of v blew up. For some reason, when t is small, the model
        has a very hard time learning a good representation of v.
        So, I am adding a restraint to keep it between 0 and 1.
        """
        v = v.sigmoid()
        
        # Return the variance value
        return torch.exp(torch.clamp(v*torch.log(beta_t) + (1-v)*torch.log(beta_tilde_t), torch.tensor(-30, device=beta_t.device), torch.tensor(30, device=beta_t.device)))
        
        
        
    # Input:
    #   x_t - Batch of images of shape (B, C, L, W)
    #   t - (Optional) Batch of t values of shape (N) or a single t value
    # Outputs:
    #   noise - Batch of noise predictions of shape (B, C, L, W)
    #   v - Batch of v predictions of shape (B, C, L, W)
    def forward(self, x_t, t):
        
        # Make sure t is in the correct form
        if t != None:
            if type(t) == int or type(t) == float:
                t = torch.tensor(t).repeat(x_t.shape[0]).to(torch.long)
            elif type(t) == list and type(t[0]) == int:
                t = torch.tensor(t).to(torch.long)
            elif type(t) == torch.Tensor:
                if len(t.shape) == 0:
                    t = t.repeat(x_t.shape[0]).to(torch.long)
            else:
                print(f"t values must either be a scalar, list of scalars, or a tensor of scalars, not type: {type(t)}")
                return
            
            # Encode the timesteps
            if len(t.shape) == 1:
                t = self.t_emb(t)
        
        # Send the input through the U-net to get
        # the model output
        out = self.unet(x_t, t)

        # Get the noise and v predictions
        # for the image x_t-1
        noise, v = out[:, self.inCh:], out[:, :self.inCh]

        # Send the predictions through a convolution layer
        noise = self.out_mean(noise)
        v = self.out_var(v)
        
        return noise, v
    
    
    
    # Given the mean, variance, and input for a normal distribution,
    # return the output value of the input in the distribution
    # Inputs:
    #   x - Input into the distribution
    #   mean - Distribution mean
    #   var - Distribution variance
    #   - Note: x, mean, and var should have the same shape
    # Outputs:
    #   Distribution applied to x of the same shape as x
    def normal_dist(self, x, mean, var):
        var = torch.where(torch.logical_and(var<1e-5, var>=0), var+1e-5, var)
        var = torch.where(torch.logical_and(var>-1e-5, var<0), var-1e-5, var)
        return (1/(var*torch.sqrt(torch.tensor(2*torch.pi))))*\
            torch.exp((-1/2)*((x-mean)/var)**2)
    
    
    
    # Given a batch of images, unoise them using the current models's state
    # Inputs:
    #   x_t - Batch of images at the given value of t of shape (N, C, L, W)
    #   t - Batch of t values of shape (N) or a single t value
    # Outputs:
    #   Image of shape (N, C, L, W) at timestep t-1, unnoised by one timestep
    def unnoise_batch(self, x_t, t):
        
        # # Scale the image to (-1, 1)
        # if x_t.max() <= 1.0:
        #     x_t = reduce_image(x_t)
        
        # Make sure t is in the correct form
        if type(t) == int or type(t) == float:
            t = torch.tensor(t).repeat(x_t.shape[0]).to(torch.long)
        elif type(t) == list and type(t[0]) == int:
            t = torch.tensor(t).to(torch.long)
        elif type(t) == torch.Tensor:
            if len(t.shape) == 0:
                t = t.repeat(x_t.shape[0]).to(torch.long)
        else:
            print(f"t values must either be a scalar, list of scalars, or a tensor of scalars, not type: {type(t)}")
            return
        
        x_t = x_t.to(self.device)
        t = t.to(self.device)
        
        # Get the model predictions for the noise and v values
        noise_t, v_t = self.forward(x_t, t)
        
        # Convert the noise to a mean
        mean_t = self.noise_to_mean(noise_t, x_t, t)

        # Convert the v prediction variance
        var_t = self.vs_to_variance(v_t, t)
        
        # Get the beta t value
        beta_t, _, _ = self.get_scheduler_info(t)
        beta_t = self.unsqueeze(beta_t, -1, 3)
        t = self.unsqueeze(t, -1, 3)
        
        # Get the output of the predicted normal distribution
        # out = self.normal_dist(x_t, mean_t, var_t)
        out = torch.where(t > 0,
            mean_t + torch.randn((mean_t.shape), device=self.device)*torch.sqrt(var_t),
            mean_t
        )
        
        # Return the image scaled to (0, 255)
        # return unreduce_image(out)
        return out
    
    def saveModel_T(self, saveDir, epoch=None):
        if epoch:
            saveFile = f"model_{epoch}.pkl"
            saveDefFile = f"model_params_{epoch}.json"
        else:
            saveFile = "model.pkl"
            saveDefFile = "model_params.json"
        
        # Check if the directory exists. If it doesn't
        # create it
        if not os.path.isdir(saveDir):
            os.makedirs(saveDir)
        
        # Save the model
        torch.save(self.state_dict(), saveDir + os.sep + saveFile)

        # Save the defaults
        with open(saveDir + os.sep + saveDefFile, "w") as f:
            json.dump(self.defaults, f)
    
    # Save the model
    def saveModel(self, saveDir, epoch=None):
        # Async saving
        thr = threading.Thread(target=self.saveModel_T, args=(saveDir, epoch), kwargs={})
        thr.start()
    
    
    # Load the model
    def loadModel(self, loadDir, loadFile, loadDefFile=None):
        if loadDefFile:
            # Load in the defaults
            with open(loadDir + os.sep + loadDefFile, "r") as f:
                self.defaults = json.load(f)
            D = self.defaults

            # Reinitialize the model with the new defaults
            self.__init__(D["inCh"], D["embCh"], D["chMult"], D["num_heads"], D["num_res_blocks"], D["T"], D["beta_sched"], D["t_dim"], self.dev, bool(D["useDeep"]))

            # Load the model state
            self.load_state_dict(torch.load(loadDir + os.sep + loadFile, map_location=self.device))

        else:
            self.load_state_dict(torch.load(loadDir + os.sep + loadFile, map_location=self.device))