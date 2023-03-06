# Realtive import
import sys
sys.path.append('../blocks')

import torch
from torch import nn
try:
    from blocks.unetBlock import unetBlock
    from blocks.Efficient_Channel_Attention import Efficient_Channel_Attention
    from blocks.Multihead_Attn import Multihead_Attn
except ModuleNotFoundError:
    from ..blocks.unetBlock import unetBlock
    from ..blocks.Efficient_Channel_Attention import Efficient_Channel_Attention
    from ..blocks.Multihead_Attn import Multihead_Attn








class U_Net(nn.Module):
    # inCh - Number of input channels in the input batch
    # outCh - Number of output channels in the output batch
    # embCh - Number of channels to embed the batch to
    # chMult - Multiplier to scale the number of channels by
    #          for each up/down sampling block
    # t_dim - Vector size for the supplied t vector
    # num_blocks - Number of blocks on the up/down path
    # blk_types - How should the residual block be structured 
    #             (list of "res", "conv", "clsAtn", and/or "chnAtn". 
    #              Ex: ["res", "res", "conv", "clsAtn", "chnAtn"] 
    # c_dim - (optional) Vector size for the supplied c vectors
    # dropoutRate - Rate to apply dropout in the model
    # atn_resolution - Resolution of the attention blocks
    def __init__(self, inCh, outCh, embCh, chMult, t_dim, num_blocks, blk_types, c_dim=None, dropoutRate=0.0, atn_resolution=16):
        super(U_Net, self).__init__()

        self.c_dim = c_dim

        # Input convolution
        self.inConv = nn.Conv2d(inCh, embCh, 7, padding=3)
        
        # Downsampling
        # (N, inCh, L, W) -> (N, embCh^(chMult*num_blocks), L/(2^num_blocks), W/(2^num_blocks))
        blocks = []
        curCh = embCh
        for i in range(1, num_blocks+1):
            blocks.append(unetBlock(curCh, embCh*(2**(chMult*i)), blk_types, t_dim, c_dim, dropoutRate=dropoutRate, atn_resolution=atn_resolution))
            if i != num_blocks+1:
                blocks.append(nn.Conv2d(embCh*(2**(chMult*i)), embCh*(2**(chMult*i)), kernel_size=3, stride=2, padding=1))
            curCh = embCh*(2**(chMult*i))
        self.downBlocks = nn.Sequential(
            *blocks
        )
        
        
        # Intermediate blocks
        # (N, embCh^(chMult*num_blocks), L/(2^num_blocks), W/(2^num_blocks))
        # -> (N, embCh^(chMult*num_blocks), L/(2^num_blocks), W/(2^num_blocks))
        intermediateCh = curCh
        self.intermediate = nn.Sequential(
            # convNext(intermediateCh, intermediateCh, t_dim, dropoutRate=dropoutRate),
            unetBlock(intermediateCh, intermediateCh, blk_types, t_dim, c_dim, dropoutRate=dropoutRate, atn_resolution=atn_resolution),
            Efficient_Channel_Attention(intermediateCh),
            # convNext(intermediateCh, intermediateCh, t_dim, dropoutRate=dropoutRate)
            unetBlock(intermediateCh, intermediateCh, blk_types, t_dim, c_dim, dropoutRate=dropoutRate, atn_resolution=atn_resolution),
        )
        
        
        # Upsample
        # (N, embCh^(chMult*num_blocks), L/(2^num_blocks), W/(2^num_blocks)) -> (N, inCh, L, W)
        blocks = []
        for i in range(num_blocks, -1, -1):
            if i == 0:
                blocks.append(unetBlock(embCh*(2**(chMult*i)), embCh*(2**(chMult*i)), blk_types, t_dim, c_dim, dropoutRate=dropoutRate, atn_resolution=atn_resolution))
                blocks.append(unetBlock(embCh*(2**(chMult*i)), outCh, blk_types, t_dim, c_dim, dropoutRate=dropoutRate, atn_resolution=atn_resolution))
            else:
                blocks.append(nn.ConvTranspose2d(embCh*(2**(chMult*(i))), embCh*(2**(chMult*(i))), kernel_size=4, stride=2, padding=1))
                blocks.append(unetBlock(2*embCh*(2**(chMult*i)), embCh*(2**(chMult*(i-1))), blk_types, t_dim, c_dim, dropoutRate=dropoutRate, atn_resolution=atn_resolution))
        self.upBlocks = nn.Sequential(
            *blocks
        )
        
        # Final output block
        self.out = nn.Conv2d(outCh, outCh, 7, padding=3)

        # Down/up sample blocks
        self.downSamp = nn.AvgPool2d(2) 
        self.upSamp = nn.Upsample(scale_factor=2)

        # Time embeddings
        self.t_emb = nn.Sequential(
                nn.Linear(t_dim, t_dim),
                nn.GELU(),
                nn.Linear(t_dim, t_dim),
            )
    
    
    # Input:
    #   X - Tensor of shape (N, Ch, L, W)
    #   t - Batch of encoded t values for each 
    #       X value of shape (N, t_dim)
    #   c - (optional) Batch of encoded c values
    #       of shape (N, c_dim)
    def forward(self, X, t, c=None):
        # Class embedding assertion
        if type(c) != type(None):
            assert type(self.c_dim) != type(None), "c_dim must be specified when using class information."

        # Encode the time embeddings
        t = self.t_emb(t)

        # Saved residuals to add to the upsampling
        residuals = []

        X = self.inConv(X)
        
        # Send the input through the downsampling blocks
        # while saving the output of each one
        # for residual connections
        b = 0
        while b < len(self.downBlocks):
            X = self.downBlocks[b](X, t, c)
            residuals.append(X.clone())
            b += 1
            if b < len(self.downBlocks) and type(self.downBlocks[b]) == nn.Conv2d:
                X = self.downBlocks[b](X)
                b += 1
            
        # Reverse the residuals
        residuals = residuals[::-1]
        
        # Send the output of the downsampling block
        # through the intermediate blocks
        for b in self.intermediate:
            try:
                X = b(X, t, c)
            except TypeError:
                X = b(X)
        
        # Send the intermediate batch through the upsampling
        # block to get the original shape
        b = 0
        while b < len(self.upBlocks):
            if b < len(self.upBlocks) and type(self.upBlocks[b]) == nn.ConvTranspose2d:
                X = self.upBlocks[b](X)
                b += 1
            if len(residuals) > 0:
                X = self.upBlocks[b](torch.cat((X, residuals[0]), dim=1), t, c)
            else:
                X = self.upBlocks[b](X, t, c)
            b += 1
            residuals = residuals[1:]
        
        # Send the output through the final block
        # and return the output
        return self.out(X)