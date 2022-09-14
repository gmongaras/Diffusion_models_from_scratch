from torch import nn
import torch




# A 2-D convolution layer with a mask
class PixelCNN_Conv(nn.Module):
    
    # mask_type - "A" to use mask type A, which restricts connections
    #             to other colors which have already been predicted.
    #             Note that mask A is only for the first convolution layer
    #             "B" to use mask type B, which allows connections to
    #             predict colors in the current pixels
    # in_channels - Number of channels in the input to the Conv layer
    # out_channels - Number of channels in the output to the Conv layer
    # kernel_size - Size of the kernel in the Convolution
    # stride - Stride to move the kernel in the convolution
    # padding - Padding to add to the convolution
    def __init__(self, mask_type, in_channels, out_channels,
                 kernel_size, stride=1, padding=0):
        super(PixelCNN_Conv, self).__init__()
        
        # Create the 2D convolution layer
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, stride, padding)
        
        # Create the mask
        """
        Example mask B:
        kernel_size = 3x4
        
        mask:
        | 1  1  1  1 |
        | 1  1  1  0 |
        | 0  0  0  0 |
        
        Example mask A:
        kernel_size = 3x4
        
        mask:
        | 1  1  1  1 |
        | 1  1  0  0 |
        | 0  0  0  0 |
        """
        kernel_shape = self.conv.weight.permute(2, 3, 1, 0).shape
        self.mask = torch.zeros(kernel_shape)
        self.mask[:kernel_shape[0] // 2] = 1.0
        self.mask[kernel_shape[0] // 2, :kernel_shape[1] // 2] = 1.0
        if mask_type == "B":
            self.mask[kernel_shape[0] // 2, kernel_shape[1] // 2] = 1.0
        self.mask = self.mask.permute(3, 2, 0, 1)
            
        # ReLU blocks
        self.relu = nn.ReLU()
        
    # Inputs:
    #   X - Images of shape (N, h, n, n)
    #       - N = batch size
    #       - h = number of feature maps (RGB)
    #       - n = image size
    # Output:
    #   Images of shape (N, out_channels, n, n)
    def forward(self, X):
        # Mask the kernel
        self.conv.weight = torch.nn.Parameter(self.conv.weight * self.mask)
        
        # Return the convoluted input
        return self.relu(self.conv(X))