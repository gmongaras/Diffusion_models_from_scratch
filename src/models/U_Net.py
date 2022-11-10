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








class U_Net(nn.Module):
    # inCh - Number of input channels in the input batch
    # outCh - Number of output channels in the output batch
    # embCh - Number of channels to embed the batch to
    # chMult - Multiplier to scale the number of channels by
    #          for each up/down sampling block
    # t_dim - Vector size for the supplied t vector
    # num_heads - Number of heads in each multi-head non-local block
    # num_res_blocks - Number of residual blocks on the up/down path
    # useDeep - True to use deep residual blocks, False to use not deep residual blocks
    def __init__(self, inCh, outCh, embCh, chMult, t_dim, num_heads, num_res_blocks, useDeep=False):
        super(U_Net, self).__init__()
        
        # What type of block should be used? deep or not deep?
        if useDeep:
            self.upBlock = BigGAN_ResUp_Deep
            self.downBlock = BigGAN_ResDown_Deep
            self.resBlock = BigGAN_Res_Deep
        else:
            self.upBlock = BigGAN_ResUp
            self.downBlock = BigGAN_ResDown
            self.resBlock = BigGAN_Res

        # Input convolution
        self.inConv = nn.Conv2d(inCh, embCh, 7, padding=3)
        
        # Downsampling
        # (N, inCh, L, W) -> (N, embCh^(chMult*num_res_blocks), L/(2^num_res_blocks), W/(2^num_res_blocks))
        blocks = []
        curCh = embCh
        # for i in range(1, num_res_blocks+1):
        #     blocks.append(self.resBlock(curCh, curCh, useCls=True, cls_dim=t_dim))
        #     blocks.append(self.downBlock(curCh, embCh*(chMult*i)))
        #     blocks.append(Non_local_MH(embCh*(chMult*i), head_res=16))
        #     curCh = embCh*(chMult*i)
        for i in range(1, num_res_blocks+1):
            blocks.append(resBlock(curCh, embCh*(chMult*i), t_dim, head_res=16))
            curCh = embCh*(chMult*i)
        self.downBlocks = nn.Sequential(
            *blocks
        )
        
        
        # Intermediate blocks
        # (N, embCh^(chMult*num_res_blocks), L/(2^num_res_blocks), W/(2^num_res_blocks))
        # -> (N, embCh^(chMult*num_res_blocks), L/(2^num_res_blocks), W/(2^num_res_blocks))
        intermediateCh = embCh*(chMult*num_res_blocks)
        # self.intermediate = nn.Sequential(
        #     self.resBlock(intermediateCh, intermediateCh, useCls=True, cls_dim=t_dim),
        #     Non_local_MH(intermediateCh, head_res=16),
        #     self.resBlock(intermediateCh, intermediateCh, useCls=True, cls_dim=t_dim)
        # )
        self.intermediate = nn.Sequential(
            convNext(intermediateCh, intermediateCh, True, t_dim),
            Non_local_MH(intermediateCh, head_res=16),
            convNext(intermediateCh, intermediateCh, True, t_dim),
        )
        
        
        # Upsample
        # (N, embCh^(chMult*num_res_blocks), L/(2^num_res_blocks), W/(2^num_res_blocks)) -> (N, inCh, L, W)
        blocks = []
        # for i in range(num_res_blocks, 0, -1):
        #     blocks.append(self.resBlock(embCh*(chMult*i), embCh*(chMult*i), useCls=True, cls_dim=t_dim))
        #     # blocks.append(Non_local_MH(embCh*(chMult*i), head_res=16))
        #     if i == 1:
        #         blocks.append(self.upBlock(embCh*(chMult*i), outCh, useCls=True, cls_dim=t_dim))
        #     else:
        #         blocks.append(self.upBlock(embCh*(chMult*i), embCh*(chMult*(i-1)), useCls=True, cls_dim=t_dim))
        for i in range(num_res_blocks, 0, -1):
            if i == 1:
                blocks.append(resBlock(2*embCh*(chMult*i), outCh, t_dim, num_heads=1))
            else:
                blocks.append(resBlock(2*embCh*(chMult*i), embCh*(chMult*(i-1)), t_dim, head_res=16))
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
    #   t - (Optional) Batch of encoded t values for each 
    #       X value of shape (N, E)
    def forward(self, X, t=None):
        t = self.t_emb(t)

        # Saved residuals to add to the upsampling
        residuals = []

        X = self.inConv(X)
        
        # Send the input through the downsampling blocks
        # while saving the output of each one
        # for residual connections
        # for b in self.downSamp:
        #     try:
        #         X = b(X, t)
        #     except TypeError:
        #         X = b(X)
        #     if type(b) == self.downBlock:
        #         residuals.append(X.clone())
        for b in self.downBlocks:
            X = b(X, t)
            residuals.append(X.clone())
            if b != self.downBlocks[-1]:
                X = self.downSamp(X)
            
        # Reverse the residuals
        residuals = residuals[::-1]
        
        # Send the output of the downsampling block
        # through the intermediate blocks
        for b in self.intermediate:
            try:
                X = b(X, t)
            except TypeError:
                X = b(X)
        # X = self.intermediate(X)
        
        # Send the intermediate batch through the upsampling
        # block to get the original shape
        # if t == None:
        #     for b in self.upSamp:
        #         if len(residuals) > 0 and type(b) == self.resBlock:
        #             try:
        #                 X += residuals[0]
        #                 residuals = residuals[1:]
        #             except RuntimeError:
        #                 pass
        #         X = b(X)
        # else:
        #     for b in self.upSamp:
        #         if len(residuals) > 0 and type(b) == self.resBlock:
        #             try:
        #                 X += residuals[0]
        #                 residuals = residuals[1:]
        #             except RuntimeError:
        #                 pass
        #         try:  
        #             X = b(X, t)
        #         except TypeError:
        #             X = b(X)
        for b in self.upBlocks:
            X = b(torch.cat((X, residuals[0]), dim=1), t)
            residuals = residuals[1:]
            if b != self.upBlocks[-1]:
                X = self.upSamp(X)
        
        # Send the output through the final block
        # and return the output
        return self.out(X)