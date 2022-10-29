import torch
from torch import nn
import math



# Thanks to Hugging Face for this awesome function!
# https://huggingface.co/blog/annotated-diffusion



class PositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    # Convert time steps to embedding tensors
    # Inputs:
    #   time - Time values of shape (N)
    # Outputs:
    #   embedded time values of shape (N, dim)
    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings