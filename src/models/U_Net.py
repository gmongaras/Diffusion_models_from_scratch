import torch
from torch import nn
from ..blocks.BigGAN_ResDown import BigGAN_ResDown
from ..blocks.BigGAN_ResUp import BigGAN_ResUp
from ..blocks.BigGAN_Res import BigGAN_Res
from ..blocks.BigGAN_ResDown_Deep import BigGAN_ResDown_Deep
from ..blocks.BigGAN_ResUp_Deep import BigGAN_ResUp_Deep
from ..blocks.BigGAN_Res_Deep import BigGAN_Res_Deep
from ..blocks.Non_local_MH import Non_local_MH
from ..blocks.resBlock import resBlock
from ..blocks.convNext import convNext
from ..blocks.Efficient_Channel_Attention import Efficient_Channel_Attention








class U_Net(nn.Module):
    # inCh - Number of input channels in the input batch
    # outCh - Number of output channels in the output batch
    # embCh - Number of channels to embed the batch to
    # chMult - Multiplier to scale the number of channels by
    #          for each up/down sampling block
    # t_dim - Vector size for the supplied t vector
    # c_dim - (optional) Vector size for the supplied c vectors
    # num_res_blocks - Number of residual blocks on the up/down path
    # dropoutRate - Rate to apply dropout in the model
    def __init__(self, inCh, outCh, embCh, chMult, t_dim, c_dim, num_res_blocks, dropoutRate=0.0):
        super(U_Net, self).__init__()

        # Input convolution
        self.inConv = nn.Conv2d(inCh, embCh, 7, padding=3)
        
        # Downsampling
        # (N, inCh, L, W) -> (N, embCh^(chMult*num_res_blocks), L/(2^num_res_blocks), W/(2^num_res_blocks))
        blocks = []
        curCh = embCh
        for i in range(1, num_res_blocks+1):
            blocks.append(resBlock(curCh, embCh*(2**(chMult*i)), t_dim, c_dim, head_res=16, dropoutRate=dropoutRate))
            if i != num_res_blocks:
                blocks.append(nn.Conv2d(embCh*(2**(chMult*i)), embCh*(2**(chMult*i)), kernel_size=3, stride=2, padding=1))
            curCh = embCh*(2**(chMult*i))
        self.downBlocks = nn.Sequential(
            *blocks
        )
        
        
        # Intermediate blocks
        # (N, embCh^(chMult*num_res_blocks), L/(2^num_res_blocks), W/(2^num_res_blocks))
        # -> (N, embCh^(chMult*num_res_blocks), L/(2^num_res_blocks), W/(2^num_res_blocks))
        intermediateCh = curCh
        self.intermediate = nn.Sequential(
            convNext(intermediateCh, intermediateCh, t_dim, c_dim, dropoutRate=dropoutRate),
            Efficient_Channel_Attention(intermediateCh),
            convNext(intermediateCh, intermediateCh, t_dim, c_dim, dropoutRate=dropoutRate),
        )
        
        
        # Upsample
        # (N, embCh^(chMult*num_res_blocks), L/(2^num_res_blocks), W/(2^num_res_blocks)) -> (N, inCh, L, W)
        blocks = []
        for i in range(num_res_blocks, 0, -1):
            if i == 1:
                blocks.append(resBlock(2*embCh*(2**(chMult*i)), embCh*(2**(chMult*i)), t_dim, c_dim, num_heads=1, dropoutRate=dropoutRate))
                blocks.append(resBlock(embCh*(2**(chMult*i)), outCh, t_dim, c_dim, num_heads=1, dropoutRate=dropoutRate))
            else:
                blocks.append(resBlock(2*embCh*(2**(chMult*i)), embCh*(2**(chMult*(i-1))), t_dim, c_dim, head_res=16, dropoutRate=dropoutRate))
                blocks.append(nn.ConvTranspose2d(embCh*(2**(chMult*(i-1))), embCh*(2**(chMult*(i-1))), kernel_size=4, stride=2, padding=1))
        self.upBlocks = nn.Sequential(
            *blocks
        )
        
        # Final output block
        self.out = nn.Conv2d(outCh, outCh, 7, padding=3)

        # Down/up sample blocks
        self.downSamp = nn.AvgPool2d(2) 
        self.upSamp = nn.Upsample(scale_factor=2)

        self.t_emb = nn.Sequential(
                nn.Linear(t_dim, t_dim),
                nn.GELU(),
                nn.Linear(t_dim, t_dim),
            )
    
    
    # Input:
    #   X - Tensor of shape (N, Ch, L, W)
    #   t - Batch of encoded t values for each 
    #       X value of shape (N, t_dim)
    #   c - (optional) Batch of encoded c values for
    #       each C value of shape (N, c_dim)
    def forward(self, X, t, c=None):
        # Emcode the time embeddings
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
                X = b(X, t)
            except TypeError:
                X = b(X)
        
        # Send the intermediate batch through the upsampling
        # block to get the original shape
        b = 0
        while b < len(self.upBlocks):
            if len(residuals) > 0:
                X = self.upBlocks[b](torch.cat((X, residuals[0]), dim=1), t, c)
            else:
                X = self.upBlocks[b](X, t, c)
            b += 1
            if b < len(self.upBlocks) and type(self.upBlocks[b]) == nn.ConvTranspose2d:
                X = self.upBlocks[b](X)
                b += 1
            residuals = residuals[1:]
        
        # Send the output through the final block
        # and return the output
        return self.out(X)