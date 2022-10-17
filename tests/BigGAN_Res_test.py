# Path hack for relative paths
import sys, os
sys.path.insert(0, os.path.abspath('./src'))

from src.blocks.BigGAN_ResUp import BigGAN_ResUp
from src.blocks.BigGAN_ResDown import BigGAN_ResDown
from src.blocks.Non_local import Non_local
import torch





def test():
    # Batch of images of shape (N, C, L, W) where
    # N = 2
    # C = 5
    # L = 10
    # W = 10
    N = 2
    C = 6
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
    
    # Non-local block (N, C, L, W) -> (N, C, L, W)
    NL = Non_local(C)
    
    # Feed the input through the non-local block
    X = NL(X)
    
    # Output should be (N, C, L, W)
    assert X.shape == (N, C, L, W)
    
    
    
if __name__ == "__main__":
    test()