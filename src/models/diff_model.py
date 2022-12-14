import torch
from torch import nn
from .U_Net import U_Net
from ..helpers.image_rescale import reduce_image, unreduce_image
from ..blocks.PositionalEncoding import PositionalEncoding
import os
import json
import threading
from ..blocks.convNext import convNext
from .Variance_Scheduler import Variance_Scheduler, DDIM_Scheduler
from tqdm import tqdm






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
    # useDeep - Use a deep version of the model or a shallow version
    # dropoutRate - Rate to apply dropout to the U-net model
    # step_size - Step size to take when generating images. This is 1 for
    #        normal generation, but a greater integer for faster generation.
    #        Note: This is not used for training
    # DDIM_scale - Scale to transition between a DDIM, DDPM, or in between.
    #              use 0 for pure DDIM and 1 for pure DDPM.
    #              Note: This is not used for training
    def __init__(self, inCh, embCh, chMult, num_heads, num_res_blocks,
                 T, beta_sched, t_dim, device, useDeep=False, dropoutRate=0.0, 
                 step_size=1, DDIM_scale=0.5):
        super(diff_model, self).__init__()
        
        self.beta_sched = beta_sched
        self.inCh = inCh
        self.step_size = step_size
        self.DDIM_scale = DDIM_scale

        assert step_size > 0 and step_size <= T, "Step size must be in the range [1, T]"
        assert DDIM_scale >= 0, "DDIM scale must be greater than or equal to 0"
        
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
        self.unet = U_Net(inCh, inCh*2, embCh, chMult, t_dim, num_heads, num_res_blocks, useDeep, dropoutRate).to(device)
        
        # DDIM Variance scheduler for values of beta and alpha
        self.scheduler = DDIM_Scheduler(beta_sched, T, self.step_size, self.device)
            
        # Used to embed the values of t so the model can use it
        self.t_emb = PositionalEncoding(t_dim).to(device)

        # Output convolutions for the mean and variance
        # self.out_mean = nn.Conv2d(inCh, inCh, 3, padding=1, groups=inCh)
        # self.out_var = nn.Conv2d(inCh, inCh, 3, padding=1, groups=inCh)
        self.out_mean = convNext(inCh, inCh).to(device)
        self.out_var = convNext(inCh, inCh).to(device)
            
            
            
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
        # Sample gaussian noise
        epsilon = torch.randn_like(X, device=self.device)
        
        # The value of a_bar_t at timestep t depending on the scheduler
        sqrt_a_bar_t = self.unsqueeze(self.scheduler.sample_sqrt_a_bar_t(t), -1, 3)
        sqrt_1_minus_a_bar_t = self.unsqueeze(self.scheduler.sample_sqrt_1_minus_a_bar_t(t), -1, 3)
        
        # Noise the images
        return sqrt_a_bar_t*X + sqrt_1_minus_a_bar_t*epsilon, epsilon
    
    
    
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
    #   corrected - True to calculate the corrected mean that doesn't
    #               go outside the bounds. False otherwise.
    # Outputs:
    #   A tensor of shape (N, C, L, W) representing the mean of the
    #     unnoised image
    def noise_to_mean(self, epsilon, x_t, t, corrected=True):
        # Note: Corrected function from the following:
        # https://github.com/hojonathanho/diffusion/issues/5

        
        # Get the beta and a values for the batch of t values
        beta_t = self.scheduler.sample_beta_t(t)
        a_t = self.scheduler.sample_a_t(t)
        sqrt_a_t = self.scheduler.sample_sqrt_a_t(t)
        a_bar_t = self.scheduler.sample_a_bar_t(t)
        sqrt_a_bar_t = self.scheduler.sample_sqrt_a_bar_t(t)
        sqrt_1_minus_a_bar_t = self.scheduler.sample_sqrt_1_minus_a_bar_t(t)
        a_bar_t1 = self.scheduler.sample_a_bar_t1(t)
        sqrt_a_bar_t1 = self.scheduler.sample_sqrt_a_bar_t1(t)

        # Make sure everything is in the correct shape
        beta_t = self.unsqueeze(beta_t, -1, 3)
        a_t = self.unsqueeze(a_t, -1, 3)
        sqrt_a_t = self.unsqueeze(sqrt_a_t, -1, 3)
        a_bar_t = self.unsqueeze(a_bar_t, -1, 3)
        sqrt_a_bar_t = self.unsqueeze(sqrt_a_bar_t, -1, 3)
        sqrt_1_minus_a_bar_t = self.unsqueeze(sqrt_1_minus_a_bar_t, -1, 3)
        a_bar_t1 = self.unsqueeze(a_bar_t1, -1, 3)
        sqrt_a_bar_t1 = self.unsqueeze(sqrt_a_bar_t1, -1, 3)
        if len(t.shape) == 1:
            t = self.unsqueeze(t, -1, 3)


        # Calculate the uncorrected mean
        if corrected == False:
            return (1/sqrt_a_t)*(x_t - (beta_t/sqrt_1_minus_a_bar_t)*epsilon)


        # Calculate the corrected mean and return it
        mean = torch.where(t == 0,
            # When t is 0, normal without correction
            (1/sqrt_a_t)*(x_t - (beta_t/sqrt_1_minus_a_bar_t)*epsilon),

            # When t is not 0, special with correction
            (sqrt_a_bar_t1*beta_t)/(1-a_bar_t) * \
                torch.clamp( (1/sqrt_a_bar_t)*x_t - (sqrt_1_minus_a_bar_t/sqrt_a_bar_t)*epsilon, -1, 1 ) + \
                (((1-a_bar_t1)*sqrt_a_t)/(1-a_bar_t))*x_t
        )
        return mean
    
    
    
    # Used to convert a batch of predicted v values to
    # a batch of variance predictions
    def vs_to_variance(self, v, t):

        # Get the scheduler information
        beta_t = self.unsqueeze(self.scheduler.sample_beta_t(t), -1, 3)
        beta_tilde_t = self.unsqueeze(self.scheduler.sample_beta_tilde_t(t), -1, 3)

        """
        Note: The authors claim that v stayed in the range of values
        it should without any type of restraint. I found that this was
        the case at later stages of t, but at early stages of t (from about t = 20 to t = 0),
        the value of v blew up. For some reason, when t is small, the model
        has a very hard time learning a good representation of v.
        So, I am adding a restraint to keep it between 0 and 1.
        """
        # v = v.sigmoid()
        
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
        std = torch.sqrt(var)

        std = torch.where(torch.logical_and(std<1e-5, std>=0), std+1e-5, std)
        std = torch.where(torch.logical_and(std>-1e-5, std<0), std-1e-5, std)
        return (1/(std*torch.sqrt(torch.tensor(2*torch.pi))))\
            * torch.exp((-1/2)*((x-mean)/std)**2) \
            + 1e-10
    
    
    
    # Given a batch of images, unoise them using the current models's state
    # Inputs:
    #   x_t - Batch of images at the given value of t of shape (N, C, L, W)
    #   t - Batch of t values of shape (N) or a single t value
    # Outputs:
    #   Image of shape (N, C, L, W) at timestep t-1, unnoised by one timestep
    def unnoise_batch(self, x_t, t):
        # Put the model in eval mode
        self.eval()
        
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

        # t is currently in DDIM state. Convert it to DDPM state
        # which is what the model's trained on to get the
        # correct noise and v prediction.
        t_enc = t*self.step_size + 1
        
        # Get the model predictions for the noise and v values
        noise_t, v_t = self.forward(x_t, t_enc)

        # Convert the v prediction variance
        var_t = self.vs_to_variance(v_t, t)






        # Variance for a DDPM to a DDIM
        # Get the beta and a values for the batch of t values
        beta_t = self.scheduler.sample_beta_t(t)
        a_t = self.scheduler.sample_a_t(t)
        sqrt_a_t = self.scheduler.sample_sqrt_a_t(t)
        a_bar_t = self.scheduler.sample_a_bar_t(t)
        sqrt_a_bar_t = self.scheduler.sample_sqrt_a_bar_t(t)
        sqrt_1_minus_a_bar_t = self.scheduler.sample_sqrt_1_minus_a_bar_t(t)
        a_bar_t1 = self.scheduler.sample_a_bar_t1(t)
        sqrt_a_bar_t1 = self.scheduler.sample_sqrt_a_bar_t1(t)
        beta_tilde_t = self.scheduler.sample_beta_tilde_t(t)

        # Make sure everything is in the correct shape
        beta_t = self.unsqueeze(beta_t, -1, 3)
        a_t = self.unsqueeze(a_t, -1, 3)
        sqrt_a_t = self.unsqueeze(sqrt_a_t, -1, 3)
        a_bar_t = self.unsqueeze(a_bar_t, -1, 3)
        sqrt_a_bar_t = self.unsqueeze(sqrt_a_bar_t, -1, 3)
        sqrt_1_minus_a_bar_t = self.unsqueeze(sqrt_1_minus_a_bar_t, -1, 3)
        a_bar_t1 = self.unsqueeze(a_bar_t1, -1, 3)
        sqrt_a_bar_t1 = self.unsqueeze(sqrt_a_bar_t1, -1, 3)
        beta_tilde_t = self.unsqueeze(beta_tilde_t, -1, 3)



        ### This is the DDIM process. x_0 blows up if not restricted, so x_0
        ### is constrained between -1.5 and 1.5 similar to the DDPM implementation.
        # The variance the model predicted and the
        # variance the model did not predict
        var_t = self.DDIM_scale*var_t
        beta_tilde_t = self.DDIM_scale*beta_tilde_t

        # Sometimes the noise blows up. Since the noise prediction should follow a normal
        # distribution, sample a normal distribution of the same shape and restrict the
        # predicted noise to the min and max of that distribution.
        samp = torch.randn_like(noise_t)
        noise_t = noise_t.clamp(samp.min(), samp.max())

        # Get the x_0 and x_t_dir predictions. Note that the
        # x_t direction prediction uses the beta_tilde_t value
        # as this value makes the process a DDPM when the scale is 1
        # but if the predicted variance is used, this isn't necessarily true.
        # The predicted variance is used as in the improved DDPM paper
        x_0_pred = ((x_t-sqrt_1_minus_a_bar_t*noise_t)/sqrt_a_bar_t)
        # x_0_pred = x_0_pred.clamp(-1.5, 1.5)
        x_t_dir_pred = torch.sqrt(torch.clamp(1-a_bar_t1-beta_tilde_t, 0, torch.inf))*noise_t
        random_noise = torch.randn((noise_t.shape), device=self.device)*torch.sqrt(var_t)

        # Get the output image for this step
        out = sqrt_a_bar_t1*x_0_pred \
            + x_t_dir_pred \
            + random_noise







        
        # Return the images
        if torch.any(torch.isnan(out)):
            print("Issue generating image. Image generation process generated nan values.")
            exit()
        return out



    # Sample a batch of generated samples from the model
    @torch.no_grad()
    def sample_imgs(self, batchSize, save_intermediate=False, use_tqdm=False, unreduce=False):
        # Make sure the model is in eval mode
        self.eval()

        # The initial image is pure noise
        output = torch.randn((batchSize, 3, 64, 64)).to(self.device)

        # Iterate T//step_size times to denoise the images
        imgs = []
        for t in tqdm(range(self.T, 0, -self.step_size)) if use_tqdm else range(self.T, 0, -self.step_size):
            output = self.unnoise_batch(output, (t//self.step_size)-1)
            if save_intermediate:
                imgs.append(unreduce_image(output[0]).cpu().detach().int().clamp(0, 255).permute(1, 2, 0))
        
        if unreduce:
            output = unreduce_image(output).clamp(0, 255)

        # Return the output images and potential intermediate output
        return (output,imgs) if save_intermediate else output


    
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
            self.__init__(D["inCh"], D["embCh"], D["chMult"], D["num_heads"], D["num_res_blocks"], D["T"], D["beta_sched"], D["t_dim"], self.dev, bool(D["useDeep"]), step_size=self.step_size, DDIM_scale=self.DDIM_scale)

            # Load the model state
            self.load_state_dict(torch.load(loadDir + os.sep + loadFile, map_location=self.device))

        else:
            self.load_state_dict(torch.load(loadDir + os.sep + loadFile, map_location=self.device))