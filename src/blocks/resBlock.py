from torch import nn
from .convNext import convNext
from .Non_local_MH import Non_local_MH
from .Multihead_Attn import Multihead_Attn
from .Efficient_Channel_Attention import Efficient_Channel_Attention
from .Spatial_Channel_Attention import Spatial_Channel_Attention
from .clsAttn import clsAttn, clsAttn_Linear, Efficient_Cls_Attention
from .wideResNet import ResnetBlock







class resBlock(nn.Module):
    # Inputs:
    #   inCh - Number of channels the input batch has
    #   outCh - Number of chanels the ouput batch should have
    #   t_dim - (optional) Number of dimensions in the time input embedding
    #   c_dim - (optional) Number of dimensions in the class embedding input
    #   num_heads - Number of heads in the multi-head
    #               non-local block
    #   head_res - Optional parameter. Specify the resolution each
    #              head operates at rather than the number of heads. If
    #              this is not None, num_heads is ignored
    #   dropoutRate - Rate to apply dropout in the convnext blocks
    #   use_attn - Should attention be used or not?
    def __init__(self, inCh, outCh, t_dim=None, c_dim=None, num_heads=2, head_res=None, dropoutRate=0.0, use_attn=True):
        super(resBlock, self).__init__()

        self.useCls = False if c_dim == None else True

        self.block = nn.Sequential(
            # convNext(inCh, outCh, t_dim, c_dim, dropoutRate),
            # convNext(outCh, outCh, t_dim, c_dim, dropoutRate),
            # clsAttn(c_dim, outCh) if self.useCls == True else nn.Identity(),
            # Efficient_Channel_Attention(outCh) if use_attn else nn.Identity(),

            ResnetBlock(inCh, outCh, t_dim, c_dim, dropoutRate),
            # ResnetBlock(outCh, outCh, t_dim, c_dim, dropoutRate),
            # convNext(outCh, outCh, t_dim, c_dim, dropoutRate),
            ResnetBlock(outCh, outCh, t_dim, c_dim, dropoutRate),
            clsAttn(c_dim, outCh) if self.useCls == True else nn.Identity(),
            Efficient_Channel_Attention(outCh) if use_attn else nn.Identity(),


            # Multihead_Attn(outCh, 2, 16, True),
            # convNext(outCh, outCh, True, t_dim, dropoutRate),
            # Non_local_MH(outCh, num_heads, head_res, spatial=True),
            # Spatial_Channel_Attention(outCh, num_heads, head_res)
        )


    # Input:
    #   X - Tensor of shape (N, inCh, L, W)
    #   t - (optional) Tensor of shape (N, t_dim)
    #   c - (optional) Tensor of shape (N, c_dim)
    # Output:
    #   Tensor of shape (N, outCh, L, W)
    def forward(self, X, t=None, c=None):
        # Class assertion
        if c != None:
            assert self.useCls == True, \
                "c_dim cannot be None if using class embeddings"

        for b in self.block:
            if type(b) == convNext or type(b) == ResnetBlock:
                X = b(X, t, c)
            elif type(b) == clsAttn or type(b) == clsAttn_Linear or type(b) == Efficient_Cls_Attention:
                X = b(X, c)
            else:
                X = b(X)
        return X