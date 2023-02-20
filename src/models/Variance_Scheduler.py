import torch






class DDIM_Scheduler():
    # sched_type - Scheduler type. Can be either "cosine" or "linear"
    # T - Maximum value of t to consider. The range will be [1, T]
    # step - Step size when generating the sequence of values to
    #        skip steps in the generation process
    # device - Device to return tensors on
    def __init__(self, sched_type, T, step, device):
        # Save the device
        self.device = device

        # Linspace for all values of t as integers
        t_vals = torch.arange(1, T+1, step).to(torch.int)

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
            self.a_bar_t1 = f((t_vals-step).clamp(0, torch.inf))

            # beta_t and alpha_t are defined from a_bar_t
            self.beta_t = 1-(self.a_bar_t/self.a_bar_t1)
            self.beta_t = torch.clamp(self.beta_t, 1e-10, 0.999)
            self.a_t = 1-self.beta_t
        else: # Linear
            
            # beta_t is defined as a linspace from 1e-4 to 0.02
            self.beta_t = torch.linspace(1e-4, 0.02, T)

            # beta_t subsequence
            self.beta_t = self.beta_t[::step]

            # alpha and alpha_bar_t are defined from the betas
            self.a_t = 1-self.beta_t
            self.a_bar_t = torch.stack([torch.prod(self.a_t[:i]) for i in range(1, (T//step)+1)])
            self.a_bar_t1 = torch.stack([torch.prod(self.a_t[:i]) for i in range(1, (T//step)+1)])
        

        # Roots of a and a_bar
        self.sqrt_a_t = torch.sqrt(self.a_t)
        self.sqrt_a_bar_t = torch.sqrt(self.a_bar_t)
        self.sqrt_1_minus_a_bar_t = torch.sqrt(1-self.a_bar_t)
        self.sqrt_a_bar_t1 = torch.sqrt(self.a_bar_t1)

        # Beta tilde value
        self.beta_tilde_t = ((1-self.a_bar_t1)/(1-self.a_bar_t))*self.beta_t


        # Move the tensors to the correct device
        self.beta_t = self.beta_t.to(self.device)
        self.a_t = self.a_t.to(self.device)
        self.a_bar_t = self.a_bar_t.to(self.device)
        self.a_bar_t1 = self.a_bar_t1.to(self.device)
        self.sqrt_a_t = self.sqrt_a_t.to(self.device)
        self.sqrt_a_bar_t = self.sqrt_a_bar_t.to(self.device)
        self.sqrt_1_minus_a_bar_t = self.sqrt_1_minus_a_bar_t.to(self.device)
        self.sqrt_a_bar_t1 = self.sqrt_a_bar_t1.to(self.device)
        self.beta_tilde_t = self.beta_tilde_t.to(self.device)

        # Unsqueeze the data to be of shape (T, 1, 1, 1) which
        # is the number of dimensions an image has (N, C, L, W)
        self.beta_t = self.beta_t.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        self.a_t = self.a_t.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        self.a_bar_t = self.a_bar_t.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        self.a_bar_t1 = self.a_bar_t1.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        self.sqrt_a_t = self.sqrt_a_t.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        self.sqrt_a_bar_t = self.sqrt_a_bar_t.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        self.sqrt_1_minus_a_bar_t = self.sqrt_1_minus_a_bar_t.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        self.sqrt_a_bar_t1 = self.sqrt_a_bar_t1.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        self.beta_tilde_t = self.beta_tilde_t.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)



    # Sampling methods. Note: Since the min t
    # is 1, 1 is subtracted to index at 0
    def sample_a_t(self, t):
        return self.a_t[t-1]
    def sample_beta_t(self, t):
        return self.beta_t[t-1]
    def sample_a_bar_t(self, t):
        return self.a_bar_t[t-1]
    def sample_a_bar_t1(self, t):
        return self.a_bar_t1[t-1]
    def sample_sqrt_a_t(self, t):
        return self.sqrt_a_t[t-1]
    def sample_sqrt_a_bar_t(self, t):
        return self.sqrt_a_bar_t[t-1]
    def sample_sqrt_1_minus_a_bar_t(self, t):
        return self.sqrt_1_minus_a_bar_t[t-1]
    def sample_sqrt_a_bar_t1(self, t):
        return self.sqrt_a_bar_t1[t-1]
    def sample_beta_tilde_t(self, t):
        return self.beta_tilde_t[t-1]