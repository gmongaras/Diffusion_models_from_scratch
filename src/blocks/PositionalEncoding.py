import torch
from torch import nn
import math



# Thanks to Hugging Face for this awesome function!
# https://huggingface.co/blog/annotated-diffusion



class PositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        # Calculate the denominator of the position encodings
        # as this value is constant
        self.denom = torch.tensor(10000)**\
            ((2*torch.arange(self.dim))/self.dim)

    # Convert time steps to embedding tensors
    # Inputs:
    #   time - Time values of shape (N)
    # Outputs:
    #   embedded time values of shape (N, dim)
    def forward(self, time):
        # Compute the current timestep embeddings
        embeddings = time[:, None]*self.denom[None, :].to(time.device)

        # Sin/Cos transformation for even, odd indices
        embeddings[::2] = embeddings[::2].sin()
        embeddings[1::2] = embeddings[1::2].cos()

        return embeddings

        # device = time.device
        # half_dim = self.dim // 2
        # embeddings = math.log(10000) / (half_dim - 1)
        # embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        # embeddings = time[:, None] * embeddings[None, :]
        # embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        # return embeddings