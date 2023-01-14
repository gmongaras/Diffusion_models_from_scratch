from torch import nn





class BigGAN_ResUp(nn.Module):
    # Inputs:
    #   inCh - Number of channels the input batch has
    #   outCh - Number of chanels the ouput batch should have
    #   t_dim - (optional) Number of dimensions in the time input embedding
    #   dropoutRate - Rate to apply dropout to each layer in the block
    def __init__(self, inCh, outCh, t_dim=None, dropout=0.0):
        super(BigGAN_ResUp, self).__init__()
        
        # Residual connection
        self.Res = nn.Sequential(
            nn.Upsample(scale_factor=2),             # (N, inCh, 2L, 2W)
            nn.Conv2d(inCh, outCh, 1)                # (N, outCh, 2L, 2W)
        )
        
        # Main Upsample flow (N, inCh, L, W) -> (N, outCh, 2L, 2W)
        self.BN1 = nn.GroupNorm(inCh//4 if inCh > 4 else 1, inCh)                         # (N, inCh, L, W)
        self.Act1 = nn.GELU()                                   # (N, inCh, L, W)
        self.Up = nn.Upsample(scale_factor=2)                   # (N, inCh, 2L, 2W)
        self.Conv1 = nn.Conv2d(inCh, outCh, 3, padding=1)       # (N, outCh, 2L, 2W)
        self.BN2 = nn.GroupNorm(inCh//4 if inCh > 4 else 1, outCh)                        # (N, outCh, 2L, 2W)
        self.Act2 = nn.GELU()                                   # (N, outCh, 2L, 2W)
        self.Conv2 = nn.Conv2d(outCh, outCh, 3, padding=1)      # (N, outCh, 2L, 2W)
        
        # Optional time vector applied over the channels
        self.t_dim = t_dim
        if t_dim != None:
            self.timeProj1 = nn.Linear(t_dim, inCh)
            self.timeProj2 = nn.Linear(t_dim, outCh)
    
    
    
    # Input:
    #   X - Tensor of shape (N, inCh, L, W)
    #   t - (optional) vector of shape (N, t_dim)
    # Output:
    #   Tensor of shape (N, outCh, 2L, 2W)
    def forward(self, X, t=None):
        # Residual path
        res = self.Res(X.clone())
        
        # Main path with optional class addition
        X = self.BN1(X)
        if self.t_dim != None and t != None:
            X += self.timeProj1(t).unsqueeze(-1).unsqueeze(-1)
        X = self.Act1(X)
        X = self.Up(X)
        X = self.Conv1(X)
        X = self.BN2(X)
        if self.t_dim != None and t != None:
            X += self.timeProj2(t).unsqueeze(-1).unsqueeze(-1)
        X = self.Act2(X)
        X = self.Conv2(X)
        
        # Add the residual to the output and return it
        return X + res