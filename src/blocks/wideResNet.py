import torch
from torch import nn
from einops import rearrange, reduce
from functools import partial
import torch.nn.functional as F



def exists(x):
    return x is not None




# From https://huggingface.co/blog/annotated-diffusion
class WeightStandardizedConv2d(nn.Conv2d):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, "o ... -> o 1 1 1", "mean")
        var = reduce(weight, "o ... -> o 1 1 1", partial(torch.var, unbiased=False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv2d(
            x,
            normalized_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )



class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=1):
        super().__init__()
        self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, t=None, c=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(t):
            x = x * t
        if exists(c):
            x = x + c

        x = self.act(x)
        return x



class ResnetBlock(nn.Module):
    """https://arxiv.org/abs/1512.03385"""
    def __init__(self, inCh, outCh, t_dim, c_dim=None, dropoutRate=0.0):
        super().__init__()
        self.t_mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(t_dim, outCh))
        )

        self.c_mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(t_dim, outCh))
            if exists(c_dim)
            else None
        )

        self.block1 = Block(inCh, outCh, groups=8 if inCh > 4 and inCh%8==0 and outCh%8==0 else 1)
        self.block2 = Block(outCh, outCh, groups=8 if outCh > 4 and outCh%8==0 else 1)
        self.res_conv = nn.Conv2d(inCh, outCh, 1) if inCh != outCh else nn.Identity()

    def forward(self, x, t=None, c=None):
        if exists(self.t_mlp) and exists(t):
            t = self.t_mlp(t)
            t = rearrange(t, "b c -> b c 1 1")

        if exists(self.c_mlp) and exists(c):
            c = self.c_mlp(c)
            c = rearrange(c, "b c -> b c 1 1")

        h = self.block1(x, t, c)
        h = self.block2(h)
        return h + self.res_conv(x)