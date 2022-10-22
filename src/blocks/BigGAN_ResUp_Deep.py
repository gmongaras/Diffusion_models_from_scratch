from torch import nn
from .ConditionalBatchNorm2D import ConditionalBatchNorm2d





class BigGAN_ResUp_Deep(nn.Module):
    # Inputs:
    #   inCh - Number of channels the input batch has
    #   outCh - Number of chanells the ouput batch should have
    #   useCls - Should class conditioning be used?
    #   num_cls - Number of classes to condition on
    #   cls_dim - Number of dimensions in the class token input
    def __init__(self, inCh, outCh, useCls=False, num_cls=0, cls_dim=128):
        super(BigGAN_ResUp_Deep, self).__init__()
        
        self.outCh = outCh
        self.useCls = useCls
        
        # Assertion for the use of classes
        if useCls:
            assert num_cls > 0, "Number of classes must be > 0 if using classes"
            
        # Assertion as the number of output channels must be less
        # than the number of input channels
        assert outCh <= inCh, "Number of output channels must be less than or equal to the number of input channels"
        
        # Residual connection
        self.ResUp = nn.Upsample(scale_factor=2)                    # (N, outCh, 2L, 2W)
        
        # Main Upsample flow (N, inCh, L, W) -> (N, outCh, 2L, 2W)
        if useCls:
            self.clsProj1 = nn.Linear(cls_dim, num_cls)
            self.BN1 = ConditionalBatchNorm2d(inCh, num_cls)        # (N, inCh, L, W)
        else:
            self.BN1 = nn.BatchNorm2d(inCh)                         # (N, inCh, L, W)
        self.Act1 = nn.ReLU()                                       # (N, inCh, L, W)
        self.Conv1 = nn.Conv2d(inCh, inCh//4, 1)                    # (N, inCh/4, L, W)
        
        if useCls:
            self.clsProj2 = nn.Linear(cls_dim, num_cls)
            self.BN2 = ConditionalBatchNorm2d(inCh//4, num_cls)     # (N, inCh/4, L, W)
        else:
            self.BN2 = nn.BatchNorm2d(inCh//4)                  # (N, inCh/4, L, W)
        self.Act2 = nn.ReLU()                                   # (N, inCh/4, L, W)
        self.Up = nn.Upsample(scale_factor=2)                   # (N, inCh/4, 2L, 2W)
        self.Conv2 = nn.Conv2d(inCh//4, inCh//4, 3, padding=1)  # (N, inCh/4, 2L, 2W)
        
        if useCls:
            self.clsProj3 = nn.Linear(cls_dim, num_cls)
            self.BN3 = ConditionalBatchNorm2d(inCh//4, num_cls) # (N, inCh/4, 2L, 2W)
        else:
            self.BN3 = nn.BatchNorm2d(inCh//4)                  # (N, inCh/4, 2L, 2W)
        self.Act3 = nn.ReLU()                                   # (N, inCh/4, 2L, 2W)
        self.Conv3 = nn.Conv2d(inCh//4, inCh//4, 3, padding=1)  # (N, inCh/4, 2L, 2W)
        
        if useCls:
            self.clsProj4 = nn.Linear(cls_dim, num_cls)
            self.BN4 = ConditionalBatchNorm2d(inCh//4, num_cls) # (N, inCh/4, 2L, 2W)
        else:
            self.BN4 = nn.BatchNorm2d(inCh//4)                  # (N, inCh/4, 2L, 2W)
        self.Act4 = nn.ReLU()                                   # (N, inCh/4, 2L, 2W)
        self.Conv4 = nn.Conv2d(inCh//4, outCh, 1)               # (N, outCh, 2L, 2W)
    
    
    
    # Input:
    #   X - Tensor of shape (N, inCh, L, W)
    #   cls - ...
    # Output:
    #   Tensor of shape (N, outCh, 2L, 2W)
    def forward(self, X, cls=None):
        # Residual path
        res = self.ResUp(X.clone()[:, :self.outCh])
        
        # Main path with optional class addition
        if self.useCls:
            X = self.BN1(X, self.clsProj1(cls))
        else:
            X = self.BN1(X)
        X = self.Act1(X)
        X = self.Conv1(X)
        
        if self.useCls:
            X = self.BN2(X, self.clsProj2(cls))
        else:
            X = self.BN2(X)
        X = self.Act2(X)
        X = self.Up(X)
        X = self.Conv2(X)
        
        if self.useCls:
            X = self.BN3(X, self.clsProj3(cls))
        else:
            X = self.BN3(X)
        X = self.Act3(X)
        X = self.Conv3(X)
        
        if self.useCls:
            X = self.BN4(X, self.clsProj4(cls))
        else:
            X = self.BN4(X)
        X = self.Act4(X)
        X = self.Conv4(X)
        
        # Add the residual to the output and return it
        return X + res