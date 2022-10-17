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
        # Note: Has shape (C_out, C_in, kernel_size, kernel_size)
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
        kernel_shape = self.conv.weight.shape
        self.mask = torch.zeros(kernel_shape)
        
        # All rows above the middle are 1s
        self.mask[:, :, :kernel_shape[2]//2] = 1
        
        # All values to the left of the very middle pixel,
        # on the middle row are 1s
        self.mask[:, :, kernel_shape[2]//2, :kernel_shape[3]//2] = 1
        
        # Mask B masks the middle pixel too
        if mask_type == "B":
            self.mask[:, :, kernel_shape[2]//2, kernel_shape[3]//2] = 1
            
        # ReLU blocks
        self.act = nn.ELU(inplace=False)
        
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
        return self.act(self.conv(X)) + 0