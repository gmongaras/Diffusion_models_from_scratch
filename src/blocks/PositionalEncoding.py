import torch
from torch import nn
from colorama import Fore




class PositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        # Calculate the denominator of the position encodings
        # as this value is constant
        self.denom = torch.tensor(10000)**\
            ((2*torch.arange(self.dim))/self.dim)


        print(Fore.RED + '******************************************************************************************************************')
        print(Fore.RED + '*If loading a pretrained model, uncomment lines 38/39 and comment lines 34/35 in src/blocks/PositionalEncoding.py*')
        print(Fore.RED + '*Error with indices in the initial implementaion, applying PEs on the batch.                                     *')
        print(Fore.RED + '******************************************************************************************************************')

    # Convert time steps to embedding tensors
    # Inputs:
    #   time - Time values of shape (N)
    # Outputs:
    #   embedded time values of shape (N, dim)
    def forward(self, time):
        # Compute the current timestep embeddings
        embeddings = time[:, None]*self.denom[None, :].to(time.device)

        # Sin/Cos transformation for even, odd indices
        embeddings[:, ::2] = embeddings[:, ::2].sin()
        embeddings[:, 1::2] = embeddings[:, 1::2].cos()

        # # Sin/Cos transformation for even, odd indices
        # embeddings[::2] = embeddings[::2].sin()
        # embeddings[1::2] = embeddings[1::2].cos()
        # Uncomment these ^ if loading a pretrained model

        return embeddings
