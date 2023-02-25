from torch import nn
from .convNext import convNext
from .Efficient_Channel_Attention import Efficient_Channel_Attention
from .clsAttn import clsAttn, clsAttn_Linear, Efficient_Cls_Attention
from .wideResNet import ResnetBlock
from .Multihead_Attn import Multihead_Attn






# Map from string form of a block to object form
str_to_blk = dict(
    res=ResnetBlock,
    conv=convNext,
    clsAtn=clsAttn,
    chnAtn=Efficient_Channel_Attention,
    atn=Multihead_Attn,
)



class unetBlock(nn.Module):
    # Inputs:
    #   inCh - Number of channels the input batch has
    #   outCh - Number of chanels the ouput batch should have
    # blk_types - How should the residual block be structured 
    #             (list of "res", "conv", "atn", "clsAtn", and/or "chnAtn". 
    #              Ex: ["res", "res", "conv", "clsAtn", "chnAtn"] 
    #   t_dim - (optional) Number of dimensions in the time input embedding
    #   c_dim - (optional) Number of dimensions in the class embedding input
    #   atn_resolution - (optional) Resolution for the attention ("atn") blocks if used
    #   dropoutRate - (optional) Rate to apply dropout in the convnext blocks
    def __init__(self, inCh, outCh, blk_types, t_dim=None, c_dim=None, atn_resolution=None, dropoutRate=0.0):
        super(unetBlock, self).__init__()

        self.useCls = False if c_dim == None else True

        # Generate the blocks. THe first blocks goes from inCh->outCh.
        # The rest goes from outCh->outCh
        blocks = []
        curCh = inCh
        curCh1 = outCh
        for blk in blk_types:
            if blk == "res":
                blocks.append(ResnetBlock(curCh, curCh1, t_dim, c_dim, dropoutRate))
            elif blk == "conv":
                blocks.append(convNext(curCh, curCh1, t_dim, c_dim, dropoutRate))
            elif blk == "clsAtn":
                blocks.append(clsAttn(c_dim, curCh))
            elif blk == "chnAtn":
                blocks.append(Efficient_Channel_Attention(curCh))
            elif blk == "atn":
                assert atn_resolution != None, "Resolution cannot be none when using attention"
                blocks.append(Multihead_Attn(curCh, resolution=atn_resolution, spatial=True))

            curCh = curCh1

        self.block = nn.Sequential(*blocks)


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