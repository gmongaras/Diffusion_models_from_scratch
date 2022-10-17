from torch import nn





class BigGAN_ResDown(nn.Module):
    # Inputs:
    #   inCh - Number of channels the input batch has
    #   outCh - Number of chanells the ouput batch should have
    def __init__(self, inCh, outCh):
        super(BigGAN_ResDown, self).__init__()
        
        # Residual connection
        self.Res = nn.Sequential(
            nn.Conv2d(inCh, outCh, 1),              # (N, outCh, L, W)
            nn.AvgPool2d(kernel_size=2)             # (N, outCh, L/2, W/2)
        )
        
        # Main downsample flow (N, inCh, L, W) -> (N, outCh, L/2, W/2)
        self.Down = nn.Sequential(
            nn.ReLU(),                              # (N, inCh, L, W)
            nn.Conv2d(inCh, outCh, 3, padding=1),   # (N, outCh, L, W)
            nn.ReLU(),                              # (N, outCh, L, W)
            nn.Conv2d(outCh, outCh, 3, padding=1),  # (N, outCh, L, W)
            nn.AvgPool2d(kernel_size=2)             # (N, outCh, L/2, W/2)
        )
    
    
    
    # Input:
    #   X - Tensor of shape (N, inCh, L, W)
    # Output:
    #   Tensor of shape (N, outCh, L/2, W/2)
    def forward(self, X):
        # Residual path
        res = self.Res(X.clone())
        
        # Main path
        X = self.Down(X)
        
        # Add the residual to the output and return it
        return X + res