# Path hack for relative paths
import sys, os
sys.path.insert(0, os.path.abspath('./src'))

import torch
from src.models.U_Net import U_Net





def test():
    N = 4
    L = 64
    W = 64
    inCh = 3
    embCh = 16
    scale = 2
    num_heads = 8
    res_blocks = 2
    net = U_Net(inCh, inCh, embCh, scale, num_heads, res_blocks)
    
    # Random input batch
    batch = torch.rand((N, inCh, L, W))
    batch_shape = batch.shape
    
    # Send the input through the U-net
    batch = net(batch)
    
    # The shapes should be the same
    assert batch_shape == batch.shape
    
    
    
    # Deep U-net
    del net
    net = U_Net(inCh, inCh, embCh, scale, num_heads, res_blocks, useDeep=True)
    
    # Send the input through the U-net
    batch = net(batch)
    
    # The shapes should be the same
    assert batch_shape == batch.shape
    
    
    
    
    
if __name__ == "__main__":
    test()