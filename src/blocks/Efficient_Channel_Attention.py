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

        # Create the convolution layer using the kernel size. Note:
        # not using Conv1d as it causes a warning message
        # "Grad strides do not match bucket view strides"
        # which I couldn't fix, but changing it to a conv2d
        # is the same operation, but doesn't cause the warning
        self.conv = nn.Conv2d(1, 1, [1, k], padding=[0, k//2], bias=False)

        # Pooling and sigmoid functions
        self.avgPool = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()




    # Inputs:
    #   X - Image tensor of shape (N, C, L, W)
    # Outputs:
    #   Image tensor of shape (N, C, L, W)
    def forward(self, X):
        # Pool the input tensor to a (N, C, 1, 1) tensor
        att = self.avgPool(X)

        # Reshape the input tensor to be of shape (N, 1, 1, C)
        att = att.permute(0, 2, 3, 1)

        # Compute the channel attention
        att = self.conv(att)

        # Apply the sigmoid function to the channel attention
        att = self.sigmoid(att)

        # Reshape the input tensor to be of shape (N, C, 1, 1)
        att = att.permute(0, 3, 1, 2)

        # Scale the input by the attention
        return X * att.expand_as(X)