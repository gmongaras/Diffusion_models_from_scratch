from torch import nn
from .convNext import convNext
from .Non_local_MH import Non_local_MH







import torch
import math
from inspect import isfunction
from functools import partial
from tqdm.auto import tqdm
from einops import rearrange
from torch import nn, einsum
import torch.nn.functional as F
class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )
        q = q * self.scale

        sim = einsum("b h d i, b h d j -> b h i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.to_out(out)




class resBlock(nn.Module):
    # Inputs:
    #   inCh - Number of channels the input batch has
    #   outCh - Number of chanels the ouput batch should have
    #   t_dim - Number of dimensions in the time input embedding
    #   num_heads - Number of heads in the multi-head
    #               non-local block
    #   head_res - Optional parameter. Specify the resolution each
    #              head operates at rather than the number of heads. If
    #              this is not None, num_heads is ignored
    def __init__(self, inCh, outCh, t_dim, num_heads=2, head_res=None):
        super(resBlock, self).__init__()

        self.block = nn.Sequential(
            convNext(inCh, inCh, True, t_dim),
            convNext(inCh, outCh, True, t_dim),
            Non_local_MH(outCh, num_heads, head_res),
            # Attention(outCh),
        )


    # Input:
    #   X - Tensor of shape (N, inCh, L, W)
    #   t - vector of shape (N, t_dim)
    # Output:
    #   Tensor of shape (N, outCh, L/2, W/2) if down else (N, outCh, 2L, 2W)
    def forward(self, X, t):
        for b in self.block:
            if type(b) == convNext:
                X = b(X, t)
            else:
                X = b(X)
        return X