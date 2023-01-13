import torch
from torch import nn
from ..blocks.PixelCNN_Conv import PixelCNN_Conv
from ..blocks.PixelCNN_Residual import PixelCNN_Residual






class PixelCNN(nn.Module):
    def __init__(self):
        super(PixelCNN, self).__init__()
        
        # 5 residual blocks
        resBlocks = [
            PixelCNN_Residual(128, "B") for i in range(0, 5)
        ]
        
        # 2 pixel conv layers
        pixConv = [
            PixelCNN_Conv("B", 128, 128, 1) for i in range(0, 2)
        ]
        
        self.model = nn.Sequential(
            # First layer is PixelCNN_Conv with mask A
            PixelCNN_Conv("A", 1, 128, 7, padding=3),
            
            # Next layer is the residual blocks
            *resBlocks,
            
            # Next layer are pixel conv layers
            *pixConv,
            
            # Final layer is a normal conv layer predicting
            # from all possible pixel values
            nn.Conv2d(128, 256, 1),
            nn.Softmax(dim=-1),
        )
        
        # Optimizer
        self.optim = torch.optim.Adam(self.parameters(), 0.0005)
        
        # Loss function
        self.loss_funct = nn.CrossEntropyLoss()
    
    
    
    def forward(self, X):
        return self.model(X)