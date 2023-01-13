import torch
from torch import nn
from ..blocks.PixelCNN_Conv import PixelCNN_Conv
from ..blocks.PixelCNN_Deconv import PixelCNN_Deconv
from ..blocks.PixelCNN_Residual import PixelCNN_Residual






class PixelCNN_PP(nn.Module):
    # num_filters - Number of filters in the hidden layers
    # num_res_net - Number of res nets for each
    #               res net block
    # K - Number of distributions the model will predict
    #     Note: This scales the output by a liner factor of 9
    def __init__(self, num_filters=160, num_res_net=6, K=5):
        super(PixelCNN_PP, self).__init__()
        
        # Number of channels in the output is 3*3K:
        # - First 3 represents the RGB values
        # - The 3 in 3K represents the 3 parts of 
        #   the distribution to predict
        # - The K in 3K represents the K distributions
        #   the model will predict
        C_out = 3*3*K
        
        """
        Model architechure:
          Input: 3x32x32
          
          1 conv block (A type mask)
          
          6 Resnet blocks:
            1.  Normal Resnet
                Subsampling
            2.  Normal Resnet
                Subsampling
            3.  Normal Resnet
            4.  Normal Resnet
                Upsampling
            5.  Normal Resnet
                Upsampling
            6.  Normal Resnet
        """
        
        # First convolution block
        # (3x32x32) -> (Cx32x32)
        self.first_conv = PixelCNN_Conv("A", 3, num_filters, 3, padding=1)
        
        # Downsampling 32x32 -> 8x8
        self.down_blocks = [
            # First resnet block with subsampling
            # (Cx32x32) -> (Cx16x16)
            nn.Sequential(
                *[PixelCNN_Residual(num_filters, "B") for i in range(num_res_net)],
                PixelCNN_Conv("B", num_filters, num_filters, 2, 2),
            ),
            
            # Second resnet block with subsampling
            # (Cx16x16) -> (Cx8x8)
            nn.Sequential(
                *[PixelCNN_Residual(num_filters, "B") for i in range(num_res_net)],
                PixelCNN_Conv("B", num_filters, num_filters, 2, 2),
            ),
            
            
            # Third resnet block without subsampling
            # (Cx8x8) -> (Cx8x8)
            nn.Sequential(
                *[PixelCNN_Residual(num_filters, "B") for i in range(num_res_net)],
            )
        ]
            
            
            
        ### Upsampling 8x8 -> 32x32
        self.up_blocks = [
            # Fourth resnet block with upsampling
            # (Cx8x8) -> (Cx16x16)
            nn.Sequential(
                *[PixelCNN_Residual(num_filters, "B") for i in range(num_res_net)],
                PixelCNN_Deconv("B", num_filters, num_filters, 2, 2),
            ),
            
            # Fifth resnet block with upsampling
            # (Cx16x16) -> (Cx32x32)
            nn.Sequential(
                *[PixelCNN_Residual(num_filters, "B") for i in range(num_res_net)],
                PixelCNN_Deconv("B", num_filters, num_filters, 2, 2),
            ),
            
            # Final resnet block
            # (Cx32x32) -> (Cx32x32)
            nn.Sequential(
                *[PixelCNN_Residual(num_filters, "B") for i in range(num_res_net)],
            )
            
            # Output: (Cx32x32)
        ]
        
        # Final linear layer to encode the output
        self.out_linear = nn.Linear(num_filters, C_out)
        
        # Softmax layer for the weights
        self.softmax = nn.LogSoftmax(-1)
        
        
        
        # Optimizer
        self.optim = torch.optim.Adam(self.parameters(), 0.0005)
        
    
    
    # Input:
    #   Tensor of shape (N x C_in x L x W)
    # Output:
    #   Tensor of shape (N x C_out x L x W)
    #                 = (N x (3*3K) x L x W)
    #   - The 3*3K has the following layout:
    #     - every 3K block is a different RGB channel:
    #       - Every 3K block has K sets of 3 paramters:
    #         - mu, s_inv, pi
    #     - Ex: K = 2 (3*3*2 = 18):
    #       [mu_R_1, s_inv_R_1, pi_R_1, mu_R_2, s_inv_R_2, pi_R_2,
    #        mu_G_1, s_inv_G_1, ...                  , pi_G_2,
    #        mu_B_1, s_inv_B_1, ...                  , pi_B_2]
    #     - Note: The model predicts s inverse, not s (no division by 0)
    def forward(self, X):
        # Saved states
        saved_states = []
        
        # First convolution
        X = self.first_conv(X)
        
        ### Downsampling
        for down_block in self.down_blocks:
            # Save the state for later
            saved_states.append(X.clone())
            
            # Send the input through the block
            X = down_block(X)
        
        
        ### Upsampling
        for b in range(0, len(self.up_blocks)):
            # Add the residual connection to the input
            X += saved_states[-(b+1)]
            
            # Send the input through the block
            X = self.up_blocks[b](X)
            
        # Final output linear layer
        X = self.out_linear(X.permute(0, 2, 3, 1))
        
        # Reshape the output so that the values
        # that need to be transformed can be transformed
        X_shape = X.shape
        X = X.reshape(*X.shape[:-1], 3, X.shape[-1]//3)
        
        # Exponentiate the s_inverse values
        X[:, :, :, :, 1::3] = torch.exp(X[:, :, :, :, 1::3])
        
        # Softmax the pi weights
        X[:, :, :, :, 2::3] = self.softmax(X[:, :, :, :, 2::3])
            
        return X.reshape(*X_shape).permute(0, 3, 1, 2)