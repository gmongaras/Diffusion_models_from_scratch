import torch
from torch import nn
from .ConditionalBatchNorm2D import ConditionalBatchNorm2d





class BigGAN_ResDown_Deep(nn.Module):
    # Inputs:
    #   inCh - Number of channels the input batch has
    #   outCh - Number of chanells the ouput batch should have
    def __init__(self, inCh, outCh):
        super(BigGAN_ResDown_Deep, self).__init__()
        self.inCh = inCh
        self.outCh = outCh
        
        # Number of output channels must be greater than
        # or equal to the number of input channels
        assert outCh >= inCh, "Out channels must be greater than or equal to the number of input channels"
            
        # Residual connection
        self.PoolDwn_res = nn.AvgPool2d(kernel_size=2)             # (N, inCh, L/2, W/2)
        if outCh != inCh:
            self.ConvRes = nn.Conv2d(inCh, outCh-inCh, 1)          # (N, outCh-inCh, L/2, W/2)
            
        self.mainPath = nn.Sequential(
            # Main Upsample flow (N, inCh, L, W) -> (N, outCh, L/2, W/2)
            nn.ReLU(),                                         # (N, inCh, L, W)
            nn.Conv2d(inCh, inCh*4, 1),                        # (N, inCh*4, L, W)
            
            nn.ReLU(),                                         # (N, inCh*4, L, W)
            nn.Conv2d(inCh*4, inCh*4, 3, padding=1),           # (N, inCh*4, L, W)
            
            nn.ReLU(),                                         # (N, inCh*4, L, W)
            nn.Conv2d(inCh*4, inCh*4, 3, padding=1),           # (N, inCh*4, L, W)
            
            nn.ReLU(),                                         # (N, inCh*4, L, W)
            nn.AvgPool2d(kernel_size=2),                       # (N, inCh*4, L/2, W/2)
            nn.Conv2d(inCh*4, outCh, 1)                        # (N, outCh, L/2, W/2)
        )
    
    
    
    # Input:
    #   X - Tensor of shape (N, inCh, L, W)
    # Output:
    #   Tensor of shape (N, outCh, L/2, W/2)
    def forward(self, X, cls=None):
        # Residual path
        res = self.PoolDwn_res(X)
        if self.inCh != self.outCh:
            res_conv = self.ConvRes(res)
            res = torch.cat((res, res_conv), dim=1)
        
        # Main path
        X = self.mainPath(X)
        
        # Add the residual to the output and return it
        return X + res