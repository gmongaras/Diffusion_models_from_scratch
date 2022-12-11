from torch import nn
import math








class Efficient_Channel_Attention(nn.Module):
    # Efficient channel attention based on
    # https://arxiv.org/abs/1910.03151

    # Inputs:
    #   channels - Number of channels in the input
    #   gamma, b - gamma and b parameters of the kernel size calculation
    def __init__(self, channels, gamma=2, b=1):
        super(Efficient_Channel_Attention, self).__init__()

        # Calculate the kernel size
        k = int(abs((math.log2(channels)/gamma)+(b/gamma)))
        k = k if k % 2 else k + 1

        # Create the convolution layer using the kernel size
        self.conv = nn.Conv1d(1, 1, k, padding=k//2, bias=False)

        # Pooling and sigmoid functions
        self.avgPool = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()




    # Inputs:
    #   X - Image tensor of shape (N, C, L, W)
    # Outputs:
    #   Image tensor of shape (N, C, L, W)
    def forward(self, X):
        # Save the input tensor as a residual
        res = X.clone()

        # Pool the input tensor to a (N, C, 1, 1) tensor
        X = self.avgPool(X)

        # Reshape the input tensor to be of shape (N, 1, C)
        X = X.squeeze(-1).permute(0, 2, 1)

        # Compute the channel attention
        X = self.conv(X)

        # Apply the sigmoid function to the channel attention
        X = self.sigmoid(X)

        # Reshape the input tensor to be of shape (N, C, 1, 1)
        X = X.permute(0, 2, 1).unsqueeze(-1)

        # Scale the input by the attention
        return res * X