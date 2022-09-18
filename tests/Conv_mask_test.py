# Path hack for relative paths
import sys, os
sys.path.insert(0, os.path.abspath('./src'))

from src.blocks.PixelCNN_Conv import PixelCNN_Conv
import torch





def main():
    # Create a convolution block to test
    # - Mask "A"
    # - 3 channels in (RGB)
    # - 3 channels out (RGB)
    # - 5x5 kenerl
    block = PixelCNN_Conv("A", 3, 4, 5)
    
    
    # To easily check the correct value, make all the
    # weights 1s.
    block.conv.weight = torch.nn.Parameter(torch.ones(block.conv.weight.shape) * block.mask)
    block.conv.bias = torch.nn.Parameter(torch.ones(block.conv.bias.shape))
    
    
    # Image to test 
    # (NxCxLxW)
    # (5x3x5x5)
    image = torch.ones((5,3,5,5), requires_grad=False)
    
    # Feed the image through the block
    out = block(image)
    
    # The output should be a matrix of 37s (12 ones in each kernel,
    # 3 kernels, one for each RGB + 1 bias (12*3 + 1))
    assert torch.all(out == 37.0), "Output is incorrect"
    
    
    
    
    # The B mask should have 40 values (1 more open mask spot
    # per kernel)
    block = PixelCNN_Conv("B", 3, 4, 5)
    block.conv.weight = torch.nn.Parameter(torch.ones(block.conv.weight.shape) * block.mask)
    block.conv.bias = torch.nn.Parameter(torch.ones(block.conv.bias.shape))
    image = torch.ones((5,3,5,5), requires_grad=False)
    out = block(image)
    assert torch.all(out == 40.0), "Output is incorrect"
    
    
    
    # Testing the A mask for a larger image. Output should
    # be 16s
    block = PixelCNN_Conv("B", 3, 4, 3)
    block.conv.weight = torch.nn.Parameter(torch.ones(block.conv.weight.shape) * block.mask)
    block.conv.bias = torch.nn.Parameter(torch.ones(block.conv.bias.shape))
    image = torch.ones((5,3,5,5), requires_grad=False)
    out = block(image)
    assert torch.all(out == 16.0), "Output is incorrect"
    
    
    
    
if __name__ == '__main__':
    main()