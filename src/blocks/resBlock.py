from torch import nn
from .convNext import convNext
from .Non_local_MH import Non_local_MH
from .Efficient_Channel_Attention import Efficient_Channel_Attention
from .Spatial_Channel_Attention import Spatial_Channel_Attention







class resBlock(nn.Module):
    # Inputs:
    #   inCh - Number of channels the input batch has
    #   outCh - Number of chanels the ouput batch should have
    #   t_dim - (optional) Number of dimensions in the time input embedding
    #   num_heads - Number of heads in the multi-head
    #               non-local block
    #   head_res - Optional parameter. Specify the resolution each
    #              head operates at rather than the number of heads. If
    #              this is not None, num_heads is ignored
    #   dropoutRate - Rate to apply dropout in the convnext blocks
    #   use_attn - Should attention be used or not?
    def __init__(self, inCh, outCh, t_dim=None, num_heads=2, head_res=None, dropoutRate=0.0, use_attn=True):
        super(resBlock, self).__init__()

        self.block = nn.Sequential(
            convNext(inCh, inCh, t_dim, dropoutRate),
            convNext(inCh, outCh, t_dim, dropoutRate),
            # convNext(outCh, outCh, True, t_dim, dropoutRate),
            Efficient_Channel_Attention(outCh) if use_attn else nn.Identity(),
            # Non_local_MH(outCh, num_heads, head_res, spatial=True),
            # Spatial_Channel_Attention(outCh, num_heads, head_res)
        )


    # Input:
    #   X - Tensor of shape (N, inCh, L, W)
    #   t - (optional) vector of shape (N, t_dim)
    # Output:
    #   Tensor of shape (N, outCh, L/2, W/2) if down else (N, outCh, 2L, 2W)
    def forward(self, X, t=None):
        for b in self.block:
            if type(b) == convNext:
                X = b(X, t)
            else:
                X = b(X)
        return X