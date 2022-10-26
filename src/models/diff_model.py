import torch
from torch import nn
from .U_Net import U_Net






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
    def __init__(self, inCh, embCh, chMult, num_heads, num_res_blocks,
                 T, beta_sched, useDeep=False):
        super(diff_model, self).__init__()
        
        self.T = torch.tensor(T)
        self.beta_sched = beta_sched
        self.inCh = inCh
        
        # U_net model
        self.unet = U_Net(inCh, inCh*2, embCh, chMult, num_heads, num_res_blocks, useDeep)
        
        # What scheduler should be used to add noise
        # to the data?
        if self.beta_sched == "cosine":
            def f(t):
                s = 0.008
                return torch.cos(((t/T + s)/(1+s)) * torch.pi/2)**2 /\
                    torch.cos(torch.tensor((s/(1+s)) * torch.pi/2))**2
            self.beta_sched_funct = f
        else: # Linear
            self.beta_sched_funct = torch.linspace(1e-4, 0.02, T)
        
    # Used to noise a batch of images by t timesteps
    # Inputs:
    #   X - Batch of images of shape (N, C, L, W)
    #   t - Batch of t values of shape (N)
    # Outputs:
    #   Batch of noised images of shape (N, C, L, W)
    #   Noise added to the images
    def noise_batch(self, X, t):
        # Make sure t isn't too large
        t = torch.min(t, self.T)
        
        # Sample gaussian noise
        epsilon = torch.randn_like(X)
        
        # The value of a_bar_t at timestep t depending on the scheduler
        if self.beta_sched == "cosine":
            a_t_bar = self.beta_sched_funct(t)
        else:
            a_t = 1-self.beta_sched_funct[:t] # 1-B_t
            a_t_bar = torch.prod(a_t, dim=-1) # Pi [a_s]
        
        # Noise the images
        return torch.sqrt(a_t_bar)*X + torch.sqrt(1-a_t_bar)*epsilon, epsilon
    
    
    
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
        epsilon = torch.randn(noise_shape)
        
        return epsilon
        
        
        
    # Input:
    #   x_t - Batch of images of shape (B, C, L, W)
    # Outputs:
    #   noise - Batch of noise predictions of shape (B, C, L, W)
    #   vars - Batch of variance predictions of shape (B, C, L, W)
    def forward(self, x_t):
        # Send the input through the U-net to get
        # the mean and std of the gaussian distributions
        # for the image x_t-1
        out = self.unet(x_t)
        
        # Get the noise prediction from the output
        noise = out[:, :self.inCh]
        
        # Get the variances from the model
        vars = out[:, self.inCh:]
        
        return noise, vars