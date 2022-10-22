import torch
from torch import nn
from ..blocks.BigGAN_ResDown import BigGAN_ResDown
from ..blocks.BigGAN_ResUp import BigGAN_ResUp
from ..blocks.BigGAN_Res import BigGAN_Res
from ..blocks.BigGAN_ResDown_Deep import BigGAN_ResDown_Deep
from ..blocks.BigGAN_ResUp_Deep import BigGAN_ResUp_Deep
from ..blocks.BigGAN_Res_Deep import BigGAN_Res_Deep
from ..blocks.Non_local_MH import Non_local_MH








class U_Net(nn.Module):
    # inCh - Number of input channels in the input batch
    # embCh - Number of channels to embed the batch to
    # chMult - Multiplier to scale the number of channels by
    #          for each up/down sampling block
    # num_heads - Number of heads in each multi-head non-local block
    # num_res_blocks - Number of residual blocks on the up/down path
    # useDeep - True to use deep residual blocks, False to use not deep residual blocks
    def __init__(self, inCh, embCh, chMult, num_heads, num_res_blocks, useDeep=False):
        super(U_Net, self).__init__()
        
        # What type of block should be used? deep or not deep?
        if useDeep:
            upBlock = BigGAN_ResUp_Deep
            downBlock = BigGAN_ResDown_Deep
            resBlock = BigGAN_Res_Deep
        else:
            upBlock = BigGAN_ResUp
            downBlock = BigGAN_ResDown
            resBlock = BigGAN_Res
        
        # Downsampling
        # (N, inCh, L, W) -> (N, embCh^(chMult*num_res_blocks), L/(2^num_res_blocks), W/(2^num_res_blocks))
        blocks = []
        curCh = inCh
        for i in range(1, num_res_blocks+1):
            blocks.append(downBlock(curCh, embCh*(chMult*i)))
            blocks.append(Non_local_MH(embCh*(chMult*i), num_heads))
            curCh = embCh*(chMult*i)
        self.downSamp = nn.Sequential(
            *blocks
        )
        
        
        # Intermediate blocks
        # (N, embCh^(chMult*num_res_blocks), L/(2^num_res_blocks), W/(2^num_res_blocks))
        # -> (N, embCh^(chMult*num_res_blocks), L/(2^num_res_blocks), W/(2^num_res_blocks))
        intermediateCh = embCh*(chMult*num_res_blocks)
        self.intermediate = nn.Sequential(
            resBlock(intermediateCh, intermediateCh),
            Non_local_MH(intermediateCh, num_heads),
            resBlock(intermediateCh, intermediateCh)
        )
        
        
        # Upsample
        # (N, embCh^(chMult*num_res_blocks), L/(2^num_res_blocks), W/(2^num_res_blocks)) -> (N, inCh, L, W)
        blocks = []
        for i in range(num_res_blocks, 0, -1):
            blocks.append(resBlock(embCh*(chMult*i), embCh*(chMult*i)))
            blocks.append(Non_local_MH(embCh*(chMult*i), num_heads))
            if i == 1:
                blocks.append(upBlock(embCh*(chMult*i), inCh))
            else:
                blocks.append(upBlock(embCh*(chMult*i), embCh*(chMult*(i-1))))
        self.upSamp = nn.Sequential(
            *blocks
        )
    
    
    # Input:
    #   X - Tensor of shape (N, Ch, L, W)
    def forward(self, X):
        # Send the input through the downsampling blocks
        X = self.downSamp(X)
        
        # Send the output of the downsampling block
        # through the intermediate blocks
        X = self.intermediate(X)
        
        # Send the intermediate batch through the upsampling
        # block to get the original shape
        X = self.upSamp(X)
        
        return X