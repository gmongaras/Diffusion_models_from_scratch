import torch
from torch import nn
from einops import rearrange, reduce
from functools import partial
import torch.nn.functional as F



# Note, these blocks are from https://huggingface.co/blog/annotated-diffusion




# Check if something exists. Return None if it doesn't
def exists(x):
    return x is not None





# Weight standardization is shown to improve convolutions
# when using groupNorm.
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
    """
    Each block consists of:
        A weight standardized convolution (3x3)
        A group norm block
        A Silu block
    
    The original convolution was conv 3x3 -> ReLU,
    but it was found that group norm + weight standardization
    improves the performance of the model.
    """
    def __init__(self, dim, dim_out, groups=1):
        super().__init__()
        self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, t=None, c=None):
        # Project and normalize the embeddings
        x = self.proj(x)
        x = self.norm(x)

        # To add the class and time information, the
        # embedding is scaled by the time embeddings
        # and shifted by the class embeddings. In the
        # diffusion models beat gans on image synthesis paper
        # and is called Adaptive Group Normalization
        # https://arxiv.org/abs/2105.05233
        if exists(t):
            x = x * t
        if exists(c):
            x = x + c

        # Apply the SiLU layer to the embeddings
        x = self.act(x)
        return x



class ResnetBlock(nn.Module):
    """
    https://arxiv.org/abs/1512.03385
    This resnet block consits of:
        1 residual block with 8 groups (or 1 group) using cls and time info
        1 residual block with 8 groups (or 1 group) not using cls and time info
        one output convolution to project the embeddings from 
            the input channels to the output channels
    
    For the time and class embeddings, the embeddings
    is projected using a linear layer and SiLU layer
    before entering the residual block
    """
    def __init__(self, inCh, outCh, t_dim, c_dim=None, dropoutRate=0.0):
        # Projections with time and class info
        super().__init__()
        self.t_mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(t_dim, outCh))
        )
        self.c_mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(c_dim, outCh))
            if exists(c_dim)
            else None
        )

        # Convolutional blocks for residual and projections
        self.block1 = Block(inCh, outCh, groups=8 if inCh > 4 and inCh%8==0 and outCh%8==0 else 1)
        self.block2 = Block(outCh, outCh, groups=8 if outCh > 4 and outCh%8==0 else 1)
        self.res_conv = nn.Conv2d(inCh, outCh, 1) if inCh != outCh else nn.Identity()

    def forward(self, x, t=None, c=None):
        # Apply the class and time projections
        if exists(self.t_mlp) and exists(t):
            t = self.t_mlp(t)
            t = rearrange(t, "b c -> b c 1 1")
        if exists(self.c_mlp) and exists(c):
            c = self.c_mlp(c)
            c = rearrange(c, "b c -> b c 1 1")

        # Apply the convolutional blocks and
        # output projection with a residual connection
        h = self.block1(x, t, c)
        h = self.block2(h)
        return h + self.res_conv(x)