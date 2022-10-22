from torch import nn
from .ConditionalBatchNorm2D import ConditionalBatchNorm2d





# 
class BigGAN_ResUp(nn.Module):
    # Inputs:
    #   inCh - Number of channels the input batch has
    #   outCh - Number of chanells the ouput batch should have
    #   useCls - Should class conditioning be used?
    #   num_cls - Number of classes to condition on
    #   cls_dim - Number of dimensions in the class token input
    def __init__(self, inCh, outCh, useCls=False, num_cls=0, cls_dim=128):
        super(BigGAN_ResUp, self).__init__()
        
        # Assertion for the use of classes
        if useCls:
            assert num_cls > 0, "Number of classes must be > 0 if using classes"
        
        # Residual connection
        self.Res = nn.Sequential(
            nn.Upsample(scale_factor=2),             # (N, inCh, 2L, 2W)
            nn.Conv2d(inCh, outCh, 1)                # (N, outCh, 2L, 2W)
        )
        
        # Main Upsample flow (N, inCh, L, W) -> (N, outCh, 2L, 2W)
        if useCls:
            self.BN1 = ConditionalBatchNorm2d(inCh, num_cls)    # (N, inCh, L, W)
        else:
            self.BN1 = nn.BatchNorm2d(inCh)                     # (N, inCh, L, W)
        self.Act1 = nn.ReLU()                                   # (N, inCh, L, W)
        self.Up = nn.Upsample(scale_factor=2)                   # (N, inCh, 2L, 2W)
        self.Conv1 = nn.Conv2d(inCh, outCh, 3, padding=1)       # (N, outCh, 2L, 2W)
        if useCls:
            self.BN2 = ConditionalBatchNorm2d(outCh, num_cls)   # (N, outCh, 2L, 2W)
        else:
            self.BN2 = nn.BatchNorm2d(outCh)                    # (N, outCh, 2L, 2W)
        self.Act2 = nn.ReLU()                                   # (N, outCh, 2L, 2W)
        self.Conv2 = nn.Conv2d(outCh, outCh, 3, padding=1)      # (N, outCh, 2L, 2W)
        
        # Optional class token
        self.useCls = useCls
        if useCls:
            self.clsProj1 = nn.Linear(cls_dim, num_cls)
            self.clsProj2 = nn.Linear(cls_dim, num_cls)
    
    
    
    # Input:
    #   X - Tensor of shape (N, inCh, L, W)
    #   cls - ...
    # Output:
    #   Tensor of shape (N, outCh, 2L, 2W)
    def forward(self, X, cls=None):
        # Residual path
        res = self.Res(X.clone())
        
        # Main path with optional class addition
        if self.useCls:
            X = self.BN1(X, self.clsProj1(cls))
        else:
            X = self.BN1(X)
        X = self.Act1(X)
        X = self.Up(X)
        X = self.Conv1(X)
        if self.useCls:
            X = self.BN2(X, self.clsProj2(cls))
        else:
            X = self.BN2(X)
        X = self.Act2(X)
        X = self.Conv2(X)
        
        # Add the residual to the output and return it
        return X + res