import torch
from torch import nn
from .PixelCNN_Conv import PixelCNN_Conv




### The PixelCNN residual block has the following layers:
###   1x1 Conv
###   3x3 PixelCNN_Conv
###   1x1 Conv
class PixelCNN_Residual(nn.Module):
    # in_channels - Number of channels in the input
    # mask_type - Mask type for the 3x3 conv ("A" or "B")
    def __init__(self, in_channels, mask_type, dropout=0.5):
        super(PixelCNN_Residual, self).__init__()
        
        # First 1x1 conv
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//2, 1),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Second 3x3 conv
        self.conv2 = nn.Sequential(
            PixelCNN_Conv(mask_type, in_channels//2, 
                          in_channels//2, 3, padding=1),
            
        )
        
        # Final 1x1 conv
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels//2, in_channels, 1),
            nn.ReLU(),
        )
    
    
    def forward(self, X):
        X_ = self.conv1(X)
        X_ = self.conv2(X_)
        X_ = self.conv3(X_)
        return X_ + X