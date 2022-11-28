import torch
from torch import nn








class Variance_Scheduler():
    # sched_type - Scheduler type. Can be either "cosine" or "linear"
    # T - Maximum value of t to consider. The range will be [0, T]
    # device - Device to return tensors on
    def __init__(self, sched_type, T, device):

        # Save the device
        self.device = device

        # Linspace for all values of t as integers
        t_vals = torch.linspace(0, T, T+1).to(torch.int)


        # What scheduler should be used to add noise
        # to the data? For this scheduler, define
        # the beta values.
        if sched_type == "cosine":
            def f(t):
                s = 0.008
                return torch.clamp(torch.cos(((t/T + s)/(1+s)) * (torch.pi/2))**2 /\
                    torch.cos(torch.tensor((s/(1+s)) * (torch.pi/2)))**2,
                    1e-10,
                    0.999)

            # alpha_bar_t is defined directly from the scheduler
            self.a_bar_t = f(t_vals)
            self.a_bar_t1 = f(t_vals-1)

            # beta_t and alpha_t are defined from a_bar_t
            self.beta_t = 1-(self.a_bar_t/self.a_bar_t1)
            self.beta_t = torch.clamp(self.beta_t, 1e-10, 0.999)
            self.a_t = 1-self.beta_t
        else: # Linear
            
            # beta_t is defined as a linspace from 1e-4 to 0.02
            self.beta_t = torch.linspace(1e-4, 0.02, T+1)

            # alpha and alpha_bar_t are defined from the betas
            self.a_t = 1-self.beta_t
            self.a_bar_t = torch.stack([torch.tensor(0.999)] + [torch.prod(self.a_t[:i]) for i in range(1, T+1)])
            self.a_bar_t1 = torch.stack([torch.prod(self.a_t[:i]) for i in range(1, T+1)] + [torch.tensor(1e-10)])
        

        # Roots of a and a_bar
        self.sqrt_a_t = torch.sqrt(self.a_t)
        self.sqrt_a_bar_t = torch.sqrt(self.a_bar_t)
        self.sqrt_1_minus_a_bar_t = torch.sqrt(1-self.a_bar_t)
        self.sqrt_a_bar_t1 = torch.sqrt(self.a_bar_t1)

        # Beta tilde value
        self.beta_tilde_t = ((1-self.a_bar_t1)/(1-self.a_bar_t))*self.beta_t



    # Sampling methods
    def sample_a_t(self, t):
        return self.a_t[t].to(self.device)
    def sample_beta_t(self, t):
        return self.beta_t[t].to(self.device)
    def sample_a_bar_t(self, t):
        return self.a_bar_t[t].to(self.device)
    def sample_a_bar_t1(self, t):
        return self.a_bar_t1[t].to(self.device)
    def sample_sqrt_a_t(self, t):
        return self.sqrt_a_t[t].to(self.device)
    def sample_sqrt_a_bar_t(self, t):
        return self.sqrt_a_bar_t[t].to(self.device)
    def sample_sqrt_1_minus_a_bar_t(self, t):
        return self.sqrt_1_minus_a_bar_t[t].to(self.device)
    def sample_sqrt_a_bar_t1(self, t):
        return self.sqrt_a_bar_t1[t].to(self.device)
    def sample_beta_tilde_t(self, t):
        return self.beta_tilde_t[t].to(self.device)