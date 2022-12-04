import torch
from torch import nn
from .Efficient_Channel_Attention import Efficient_Channel_Attention
from .Non_local_MH import Non_local_MH












class Spatial_Channel_Attention(nn.Module):
    # Inputs:
    #   channels - Number of input channels
    #   num_heads - Number of heads in the multi-head
    #               non-local block
    #   head_res - Optional parameter. Specify the resolution each
    #              head operates at rather than the number of heads. If
    #              this is not None, num_heads is ignored
    def __init__(self, channels, num_heads=2, head_res=None):
        super(Spatial_Channel_Attention, self).__init__()

        # This block need spatial and channel attention
        self.spatial = Non_local_MH(channels, num_heads, head_res, True)
        self.channel = Efficient_Channel_Attention(channels)

        # Convolution to change the output of the attention mechanisms
        self.out = nn.Sequential(
            nn.Conv2d(channels*2, channels, 1),
        )

    

    # Forward-feed
    # Inputs:
    #   X - Images of shape (N, C, L, W)
    # Outputs:
    #   Images of shape (N, C, L, W)
    def forward(self, X):
        # Send the data through the attention mechanisms
        # and combine thte outputs
        X = torch.cat((self.spatial(X), self.channel(X)), dim=1)

        # Send the output of the attention through the
        # convolution blocks
        X = self.out(X)

        return X