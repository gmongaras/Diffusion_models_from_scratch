# Path hack for relative paths
import sys, os
sys.path.insert(0, os.path.abspath('./src'))

from src.blocks.BigGAN_ResUp import BigGAN_ResUp
from src.blocks.BigGAN_Res import BigGAN_Res
from src.blocks.BigGAN_Res_Deep import BigGAN_Res_Deep
from src.blocks.BigGAN_ResUp_Deep import BigGAN_ResUp_Deep
from src.blocks.BigGAN_ResDown_Deep import BigGAN_ResDown_Deep
from src.blocks.BigGAN_ResDown import BigGAN_ResDown
from src.blocks.Non_local import Non_local
from src.blocks.Non_local_MH import Non_local_MH
import torch





def test():
    # Batch of images of shape (N, C, L, W) where
    # N = 2
    # C = 5
    # L = 10
    # W = 10
    N = 2
    C = 20
    L = W = 10
    X = torch.rand((N, C, L, W))
    
    # Upsampling block (N, C, L, W) -> (N, 2C, 2L, 2W)
    Up = BigGAN_ResUp(C, 2*C)
    
    # Feed the input through the Upsampling block
    X = Up(X)
    
    # Output should be (N, 2C, 2L, 2W)
    assert X.shape == (N, 2*C, 2*L, 2*W)
    
    # Downsampling block (N, 2C, 2L, 2W) -> (N, C, L, W)
    Down = BigGAN_ResDown(2*C, C)
    
    # Feed the input through the Downsampling block
    X = Down(X)
    
    # Output should be (N, C, L, W)
    assert X.shape == (N, C, L, W)
    
    # Residual block (N, C, L, W) -> (N, C, L, W)
    Res = BigGAN_Res(C, C)
    
    # Feed the input through the Residual block
    X = Res(X)
    
    # Output should be (N, C, L, W)
    assert X.shape == (N, C, L, W)
    
    # Non-local block (N, C, L, W) -> (N, C, L, W)
    NL = Non_local(C)
    
    # Feed the input through the non-local block
    X = NL(X)
    
    # Output should be (N, C, L, W)
    assert X.shape == (N, C, L, W)
    
    # Multi-Head Non-local block (N, C, L, W) -> (N, C, L, W)
    NL2 = Non_local_MH(C, 5)
    
    # Feed the input through the non-local block
    X = NL2(X)
    
    # Output should be (N, C, L, W)
    assert X.shape == (N, C, L, W)
    
    # Multi-Head Non-local block using 4 resolution (N, C, L, W) -> (N, C, L, W)
    NL3 = Non_local_MH(C, head_res=4)
    
    # Feed the input through the non-local block
    X = NL3(X)
    
    # Output should be (N, C, L, W)
    assert X.shape == (N, C, L, W)
    
    
    # Deep Res Block block (N, C, L, W) -> (N, C, L, W)
    Res2 = BigGAN_Res_Deep(C, C)
    
    # Feed the input through the non-local block
    X = Res2(X)
    
    # Output should be (N, C, L, W)
    assert X.shape == (N, C, L, W)
    
    
    # Deep ResUp Block (N, C, L, W) -> (N, C/2, 2L, 2W)
    Up2 = BigGAN_ResUp_Deep(C, C//2)
    
    # Feed the input through the Deep Upsampling block
    X = Up2(X)
    
    # Output should be (N, C/2, 2L, 2W)
    assert X.shape == (N, C//2, 2*L, 2*W)
    
    
    # Deep ResDown Block (N, C/2, 2L, 2W) -> (N, C/2, L, W)
    Down2 = BigGAN_ResDown_Deep(C//2, C//2)
    
    # Feed the input through the Deep Downsampling block
    X = Down2(X)
    
    # Output should be (N, C/2, L, W)
    assert X.shape == (N, C//2, L, W)
    
    
    # Deep ResDown Block (N, C/2, L, W) -> (N, C, L/2, W/2)
    Down3 = BigGAN_ResDown_Deep(C//2, C)
    
    # Feed the input through the Deep Downsampling block
    X = Down3(X)
    
    # Output should be (N, C, L/2, W/2)
    assert X.shape == (N, C, L//2, W//2)
    
    
    
if __name__ == "__main__":
    test()