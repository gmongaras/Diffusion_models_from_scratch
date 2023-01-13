from torch import nn
from .ConditionalBatchNorm2D import ConditionalBatchNorm2d





class BigGAN_Res(nn.Module):
    # Inputs:
    #   inCh - Number of channels the input batch has
    #   outCh - Number of chanels the ouput batch should have
    #   useCls - Should class conditioning be used?
    #   cls_dim - Number of dimensions in the class token input
    def __init__(self, inCh, outCh, useCls=False, cls_dim=None):
        super(BigGAN_Res, self).__init__()
        
        if useCls == True:
            assert cls_dim != None, "Class dimenion must be supplied if the use of a class token is True"
        
        # Residual connection
        self.Res = nn.Conv2d(inCh, outCh, 1)                    # (N, outCh, L, W)
        
        # Main Upsample flow (N, inCh, L, W) -> (N, outCh, 2L, 2W)
        self.BN1 = nn.GroupNorm(4, inCh)                     # (N, inCh, L, W)
        self.Act1 = nn.ReLU()                                   # (N, inCh, L, W)
        self.Conv1 = nn.Conv2d(inCh, outCh, 3, padding=1)       # (N, outCh, L, W)
        self.BN2 = nn.GroupNorm(4, outCh)                    # (N, outCh, L, W)
        self.Act2 = nn.ReLU()                                   # (N, outCh, L, W)
        self.Conv2 = nn.Conv2d(outCh, outCh, 3, padding=1)      # (N, outCh, L, W)
        
        # Optional class token
        self.useCls = useCls
        if useCls:
            self.clsProj1 = nn.Linear(cls_dim, inCh)
            self.clsProj2 = nn.Linear(cls_dim, outCh)
    
    
    
    # Input:
    #   X - Tensor of shape (N, inCh, L, W)
    #   cls - ...
    # Output:
    #   Tensor of shape (N, outCh, L, W)
    def forward(self, X, cls=None):
        # Residual path
        res = self.Res(X.clone())
        
        # Main path with optional class addition
        X = self.BN1(X)
        if self.useCls and cls != None:
            cls1 = self.clsProj1(cls).unsqueeze(-1).unsqueeze(-1)
            X += cls1
        X = self.Act1(X)
        X = self.Conv1(X)
        if self.useCls and cls != None:
            cls2 = self.clsProj2(cls).unsqueeze(-1).unsqueeze(-1)
            X += cls2
        X = self.Act2(X)
        X = self.Conv2(X)
        
        # Add the residual to the output and return it
        return X + res