# Realtive import
import sys
sys.path.append('../helpers')
sys.path.append('../blocks')

import torch
from torch import nn
from .U_Net import U_Net
try:
    from helpers.image_rescale import reduce_image, unreduce_image
    from blocks.PositionalEncoding import PositionalEncoding
    from blocks.convNext import convNext
except ModuleNotFoundError:
    from ..helpers.image_rescale import reduce_image, unreduce_image
    from ..blocks.PositionalEncoding import PositionalEncoding
    from ..blocks.convNext import convNext
import os
import json
from .Variance_Scheduler import DDIM_Scheduler
from tqdm import tqdm






class diff_model(nn.Module):
    # inCh - Number of input channels in the input batch
    # embCh - Number of channels to embed the batch to
    # chMult - Multiplier to scale the number of channels by
    #          for each up/down sampling block
    # num_blocks - Number of blocks on the up/down path
    # blk_types - How should the residual block be structured 
    #             (list of "res", "conv", "clsAtn", and/or "chnAtn". 
    #              Ex: ["res", "res", "conv", "clsAtn", "chnAtn"] 
    #              will make unet blocks with res-res->conv->clsAtn->chnAtn)
    # T - Max number of diffusion steps
    # beta_sched - Scheduler for the beta noise term (linear or cosine)
    # useDeep - True to use deep residual blocks, False to use not deep residual blocks
    # t_dim - Embedding dimension for the timesteps
    # device - Device to put the model on (gpu or cpu)
    # c_dim - Embedding dimension for the class embeddings (None to not use class embeddings)
    # num_classes - Number of possible classes the network will work with
    # dropoutRate - Rate to apply dropout to the U-net model
    # step_size - Step size to take when generating images. This is 1 for
    #        normal generation, but a greater integer for faster generation.
    #        Note: This is not used for training
    # DDIM_scale - Scale to transition between a DDIM, DDPM, or in between.
    #              use 0 for pure DDIM and 1 for pure DDPM.
    #              Note: This is not used for training
    # start_epoch - Epoch to start on. Doesn't do much besides 
    #               change the name of the saved output file
    # start_epoch - Step to start on. Doesn't do much besides 
    #               change the name of the saved output file
    def __init__(self, inCh, embCh, chMult, num_blocks,
                 blk_types, T, beta_sched, t_dim, device, 
                 c_dim=None, num_classes=None, dropoutRate=0.0, 
                 step_size=1, DDIM_scale=0.5,
                 start_epoch=1, start_step=0):
        super(diff_model, self).__init__()
        
        self.beta_sched = beta_sched
        self.inCh = inCh
        self.step_size = step_size
        self.DDIM_scale = DDIM_scale
        self.num_classes = num_classes

        assert step_size > 0 and step_size <= T, "Step size must be in the range [1, T]"
        assert DDIM_scale >= 0, "DDIM scale must be greater than or equal to 0"
        assert (c_dim==None and num_classes==None) or \
            (c_dim!=None and num_classes!=None), \
            "c_dim and num_classes must both be specified for class information to be used"
        
        # Important default parameters
        self.defaults = {
            "inCh": inCh,
            "embCh": embCh,
            "chMult": chMult,
            "num_blocks": num_blocks,
            "blk_types": blk_types,
            "T": T,
            "beta_sched": beta_sched,
            "t_dim": t_dim,
            "c_dim": c_dim,
            "num_classes": num_classes,
            "epoch": start_epoch,
            "step": start_step
        }
        
        # Convert the device to a torch device
        if type(device) is str:
            if device.lower() == "gpu":
                if torch.cuda.is_available():
                    dev = device.lower()
                    try:
                        local_rank = int(os.environ['LOCAL_RANK'])
                    except KeyError:
                        local_rank = 0
                    device = torch.device(f"cuda:{local_rank}")
                else:
                    dev = "cpu"
                    print("GPU not available, defaulting to CPU. Please ignore this message if you do not wish to use a GPU\n")
                    device = torch.device('cpu')
            else:
                dev = "cpu"
                device = torch.device('cpu')
            self.device = device
            self.dev = dev
        else:
            self.device = device
            self.dev = "cpu" if device.type == "cpu" else "gpu"
        
        # Convert T to a tensor
        self.T = torch.tensor(T, device=device)
        
        # U_net model
        self.unet = U_Net(inCh, inCh*2, embCh, chMult, t_dim, num_blocks, blk_types, c_dim, dropoutRate).to(device)
        
        # DDIM Variance scheduler for values of beta and alpha
        self.scheduler = DDIM_Scheduler(beta_sched, T, self.step_size, self.device)
            
        # Used to embed the values of t so the model can use it
        self.t_emb = PositionalEncoding(t_dim).to(device)

        # Used to embed the values of c so the model can use it
        if c_dim != None:
            self.c_emb = nn.Linear(self.num_classes, c_dim, bias=False).to(device)
        else:
            self.c_emb = None

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
        # Ensure the data is on the correct device
        X = X.to(self.device)
        t = t.to(self.device)

        # Sample gaussian noise
        epsilon = torch.randn_like(X, device=self.device)
        
        # The value of a_bar_t at timestep t depending on the scheduler
        sqrt_a_bar_t = self.scheduler.sample_sqrt_a_bar_t(t)
        sqrt_1_minus_a_bar_t = self.scheduler.sample_sqrt_1_minus_a_bar_t(t)
        
        # Noise the images
        return sqrt_a_bar_t*X + sqrt_1_minus_a_bar_t*epsilon, epsilon



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
        sqrt_a_t = self.scheduler.sample_sqrt_a_t(t)
        a_bar_t = self.scheduler.sample_a_bar_t(t)
        sqrt_a_bar_t = self.scheduler.sample_sqrt_a_bar_t(t)
        sqrt_1_minus_a_bar_t = self.scheduler.sample_sqrt_1_minus_a_bar_t(t)
        a_bar_t1 = self.scheduler.sample_a_bar_t1(t)
        sqrt_a_bar_t1 = self.scheduler.sample_sqrt_a_bar_t1(t)


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
        beta_t = self.scheduler.sample_beta_t(t)
        beta_tilde_t = self.scheduler.sample_beta_tilde_t(t)
        
        # Return the variance value
        return torch.exp(torch.clamp(v*torch.log(beta_t) + (1-v)*torch.log(beta_tilde_t), torch.tensor(-30, device=beta_t.device), torch.tensor(30, device=beta_t.device)))
        
        
        
    # Input:
    #   x_t - Batch of images of shape (B, C, L, W)
    #   t - Batch of t values of shape (N) or a single t value. Note
    #       that this t value represents the timestep the model is currently at.
    #   c - (Optional) Batch of c values of shape (N)
    #   nullCls - (Optional) Binary tensor of shape (N) where a 1 represents a null class
    # Outputs:
    #   noise - Batch of noise predictions of shape (B, C, L, W)
    #   v - Batch of v predictions of shape (B, C, L, W)
    def forward(self, x_t, t, c=None, nullCls=None):
        # Ensure the data is on the correct device
        x_t = x_t.to(self.device)
        t = t.to(self.device)
        if c != None:
            c = c.to(self.device)
        if nullCls != None:
            nullCls = nullCls.to(self.device)

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


        # Embed the class info
        if type(c) != type(None):
            # One hot encode the class embeddings
            c = torch.nn.functional.one_hot(c.to(torch.int64), self.num_classes).to(self.device).to(torch.float)

            c = self.c_emb(c)

            # Apply the null embeddings (zeros)
            if type(nullCls) != type(None):
                c[nullCls == 1] *= 0
        
        # Send the input through the U-net to get
        # the model output
        out = self.unet(x_t, t, c)

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
    #   t_DDIM - Batch of DDIM t values of shape (N) or a single t value
    #            DDIM t values are in the range [1:T//step_size]
    #   t_DDPM - Batch of DDPM t values of shape (N) or a single t value
    #            DDPM t values are in the range [1:T]
    #   class_label - (optional and only used if the model uses class info) 
    #                 Class we want the model to generate
    #                 Use -1 to generate without a class
    #   w - (optional and only used if the model uses class info) 
    #       Classifier guidance scale factor. Use 0 for no classifier guidance.
    #   corrected - True to put a limit on generation. False to not restrain generation
    # Outputs:
    #   Image of shape (N, C, L, W) at timestep t-1, unnoised by one timestep
    def unnoise_batch(self, x_t, t_DDIM, t_DDPM, class_label=-1, w=0.0, corrected=False):
        # The model is trained on the DDPM scale while the scheduler
        # uses the DDIM scale as indices. Note that we want the model
        # to think it is at a single timestep before the timestep it generates
        # rather than the current timestep it is actually at. So if a step
        # is 20 and T is 1000, the first step starts at 981 rather than
        # 1000 to make the model think it needs to generate an image
        # from 981 to 981 rather than from 1000 to 999. So the last step
        # should be t=1 meaning it generates iamges from t=1 -> t=0. Note that the model
        # is conditioned on the timestep it is currently at, so this works.
        
        
        
        # w assertion
        assert w >= 0.0, "The value of w (classifier guidance factor) cannot be less than 0."

        # class label assertion
        class_label = int(class_label)
        assert class_label > -2 and class_label < self.num_classes,\
            f"The value of class_label must be in the range [-1,{self.num_classes-1}]"

        # Put the model in eval mode
        self.eval()
        
        # Make sure t is in the correct form
        if type(t_DDPM) == int or type(t_DDPM) == float:
            t_DDPM = torch.tensor(t_DDPM).repeat(x_t.shape[0]).to(torch.long)
        elif type(t_DDPM) == list and type(t_DDPM[0]) == int:
            t_DDPM = torch.tensor(t_DDPM).to(torch.long)
        elif type(t_DDPM) == torch.Tensor:
            if len(t_DDPM.shape) == 0:
                t_DDPM = t_DDPM.repeat(x_t.shape[0]).to(torch.long)
        else:
            print(f"t_DDPM values must either be a scalar, list of scalars, or a tensor of scalars, not type: {type(t_DDPM)}")
            return
        if type(t_DDIM) == int or type(t_DDIM) == float:
            t_DDIM = torch.tensor(t_DDIM).repeat(x_t.shape[0]).to(torch.long)
        elif type(t_DDIM) == list and type(t_DDIM[0]) == int:
            t_DDIM = torch.tensor(t_DDIM).to(torch.long)
        elif type(t_DDIM) == torch.Tensor:
            if len(t_DDIM.shape) == 0:
                t_DDIM = t_DDIM.repeat(x_t.shape[0]).to(torch.long)
        else:
            print(f"t_DDIM values must either be a scalar, list of scalars, or a tensor of scalars, not type: {type(t_DDIM)}")
            return
        
        x_t = x_t.to(self.device)
        t_DDPM = t_DDPM.to(self.device)
        t_DDIM = t_DDIM.to(self.device)
        


        ### Get the model predictions for the noise and v values

        # If the number of classes is not defined, the model
        # is not a conditioned model.
        if self.num_classes == None:
            noise_t, v_t = self.forward(x_t, t_DDPM)

        # If the number of classes is defined, the model is a
        # conditioned model
        else:
            # If the class label is -1, we only want the
            # unconditioned data
            if class_label == -1:
                noise_t, v_t = self.forward(x_t, t_DDPM, torch.tensor([0]), torch.tensor([1]))

            # If the class label is not -1, we want both
            # the conditioned and unconditioned data
            else:
                # Unconditioned sample (sample on null class)
                if w == 0:
                    noise_t_un = v_t_un = 0
                else:
                    noise_t_un, v_t_un = self.forward(x_t, t_DDPM, torch.tensor([0]), torch.tensor([1]))
                
                # Conditional sample
                noise_t_cond, v_t_cond = self.forward(x_t, t_DDPM, torch.tensor([class_label]), torch.tensor([0]))

                # Mixed sample between unconditioned and conditioned
                noise_t = (1+w)*noise_t_cond - w*noise_t_un
                v_t = (1+w)*v_t_cond - w*v_t_un

        # Convert the v prediction variance
        var_t = self.vs_to_variance(v_t, t_DDIM)






        # Variance for a DDPM to a DDIM
        # Get the beta and a values for the batch of t values
        sqrt_a_bar_t = self.scheduler.sample_sqrt_a_bar_t(t_DDIM)
        sqrt_1_minus_a_bar_t = self.scheduler.sample_sqrt_1_minus_a_bar_t(t_DDIM)
        a_bar_t1 = self.scheduler.sample_a_bar_t1(t_DDIM)
        sqrt_a_bar_t1 = self.scheduler.sample_sqrt_a_bar_t1(t_DDIM)
        beta_tilde_t = self.scheduler.sample_beta_tilde_t(t_DDIM)



        ### DDIM process

        # The variance the model predicted and the
        # variance the model did not predict
        var_t = self.DDIM_scale*var_t
        beta_tilde_t = self.DDIM_scale*beta_tilde_t

        # # Sometimes the noise blows up. Since the noise prediction should follow a normal
        # # distribution, sample a normal distribution of the same shape and restrict the
        # # predicted noise to the min and max of that distribution.
        # samp = torch.randn_like(noise_t)
        # noise_t = noise_t.clamp(samp.min(), samp.max())

        # Get the x_0 and x_t_dir predictions. Note that the
        # x_t direction prediction uses the beta_tilde_t value
        # as this value makes the process a DDPM when the scale is 1
        # but if the predicted variance is used, this isn't necessarily true.
        # The predicted variance is used as in the improved DDPM paper
        x_0_pred = ((x_t-sqrt_1_minus_a_bar_t*noise_t)/sqrt_a_bar_t)
        if corrected:
            x_0_pred = x_0_pred.clamp(-1, 1)
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
    # Params:
    #   batchSize - Number of images to generate in parallel
    #   class_label - (optional and only used if the model uses class info) 
    #                 Class we want the model to generate
    #                 Use -1 to generate without a class
    #   w - (optional and only used if the model uses class info) 
    #       Classifier guidance scale factor. Use 0 for no classifier guidance.
    #   save_intermediate - Return intermediate generation states
    #                       to create a gif along with the image?
    #   use_tqdm - Show a progress bar or not
    #   unreduce - True to unreduce the image to the range [0, 255],
    #              False to keep the image in the range [-1, 1]
    #   corrected - True to put a limit on generation. False to not restrain generation
    # Outputs:
    #   output - Output images of shape (N, C, L, W)
    #   imgs - (only if save_intermediate=True) list of iternediate
    #          outputs for the first image i the batch of shape (steps, C, L, W)
    @torch.no_grad()
    def sample_imgs(self, batchSize, class_label=-1, w=0.0, save_intermediate=False, use_tqdm=False, unreduce=False, corrected=False):
        # Make sure the model is in eval mode
        self.eval()

        # The initial image is pure noise
        output = torch.randn((batchSize, 3, 64, 64)).to(self.device)

        # Iterate T//step_size times to denoise the images (sampling from [T:1])
        imgs = []
        num_steps = len(list(reversed(range(1, self.T+1, self.step_size))))
        for t_DDIM, t_DDPM in tqdm(zip(reversed(range(1, num_steps+1)), reversed(range(1, self.T+1, self.step_size))), total=num_steps) \
            if use_tqdm else zip(reversed(range(1, num_steps+1)), reversed(range(1, self.T+1, self.step_size))):

            # Unoise by 1 step according to the DDIM and DDPM scheduler
            output = self.unnoise_batch(output, t_DDIM, t_DDPM, class_label, w, corrected)
            if save_intermediate:
                imgs.append(unreduce_image(output[0]).cpu().detach().int().clamp(0, 255).permute(1, 2, 0))
        
        # Unreduce the image from [-1:1] to [0:255]
        if unreduce:
            output = unreduce_image(output).clamp(0, 255)

        # Return the output images and potential intermediate output
        return (output,imgs) if save_intermediate else output


    
    # Save the model
    # saveDir - Directory to save the model state to
    # optimizer - Optimizer object to save the state of
    # epoch (optional) - Current epoch of the model (helps when loading state)
    # step (optional) - Current step of the model (helps when loading state)
    def saveModel(self, saveDir, optimizer, epoch=None, step=None):
        # Craft the save string
        saveFile = "model"
        saveDefFile = "model_params"
        if epoch:
            saveFile += f"_{epoch}e"
            saveDefFile += f"_{epoch}e"
        if step:
            saveFile += f"_{step}s"
            saveDefFile += f"_{step}s"
        saveFile += ".pkl"
        saveDefFile += ".json"

        # Change epoch and step state if given
        if epoch:
            self.defaults["epoch"] = epoch
        if step:
            self.defaults["step"] = step
        
        # Check if the directory exists. If it doesn't
        # create it
        if not os.path.isdir(saveDir):
            os.makedirs(saveDir)
        
        # Save the model
        torch.save(self.state_dict(), saveDir + os.sep + saveFile)

        # Save the defaults
        with open(saveDir + os.sep + saveDefFile, "w") as f:
            json.dump(self.defaults, f)
    
    
    # Load the model
    def loadModel(self, loadDir, loadFile, loadDefFile=None):
        if loadDefFile:
            # Load in the defaults
            with open(loadDir + os.sep + loadDefFile, "r") as f:
                self.defaults = json.load(f)
            D = self.defaults

            # Reinitialize the model with the new defaults
            self.__init__(D["inCh"], D["embCh"], D["chMult"], D["num_blocks"], D["blk_types"], D["T"], D["beta_sched"], D["t_dim"], self.device, D["c_dim"], D["num_classes"], 0.0, step_size=self.step_size, DDIM_scale=self.DDIM_scale, start_epoch=D["epoch"], start_step=D["step"])

            # Load the model state
            self.load_state_dict(torch.load(loadDir + os.sep + loadFile, map_location=self.device))

        else:
            self.load_state_dict(torch.load(loadDir + os.sep + loadFile, map_location=self.device))