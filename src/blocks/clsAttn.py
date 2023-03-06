import torch
from torch import nn







class clsAttn(nn.Module):
    # Inputs:
    #   cls_dim - Dimension of the class embeddings
    #   inCh - Input channels in the input embeddings
    def __init__(self, cls_dim, inCh):
        super(clsAttn, self).__init__()
        self.inCh = torch.tensor(inCh)
        
        # Query and Key embedding matrices
        self.Q_emb = nn.Linear(cls_dim, inCh)
        self.K_emb = nn.Linear(cls_dim, inCh)

        # Output embedding transformation
        self.out_emb = nn.Sequential(
            nn.Conv2d(inCh, inCh, 1),
            nn.GroupNorm(inCh//4 if inCh > 4 else 1, inCh)
        )
        
        # Layer normalization
        self.LN = nn.GroupNorm(inCh//4 if inCh > 4 else 1, inCh)

        # Attention matrix softmax
        self.softmax = nn.Softmax(-1)
        
        
    # Given an input tensor and class embeddings, attend to the
    # input tensor using the class embeddings
    # Inputs:
    #   X - tensor of shape (N, inCh, L, W) to attend to
    #   cls - tensor of shape (N, cls_emb) to make the attention with
    # Outputs:
    #   Tensor of shape (N, inCh, L, W)
    def forward(self, X, cls):
        res = X.clone()

        # Normalize input
        X = self.LN(X)

        # Get the keys and queries
        # cls: (N, cls_emb) -> (N, inCh)
        K, Q = self.Q_emb(cls), self.K_emb(cls)

        # Extend the class embeddings
        # (N, inCh) -> (N, inCh, 1)
        K = K.unsqueeze(-1)
        Q = Q.unsqueeze(-1)

        # Create the attention matrix
        # (N, inCh, 1) * (N, inCh, 1) -> (N, inCh, inCh)
        KQ = self.softmax((K@Q.permute(0, 2, 1))/torch.sqrt(self.inCh))

        # Apply the attention matrix to the input embeddings
        # (N, inCh, L, W) * (N, inCh, inCh) = (N, inCh, L, W)
        X = torch.einsum("nclw, ncd -> nclw", X, KQ)

        # Output embeddings
        X = self.out_emb(X)
        
        # Return the output with the input as a residual
        return X + res





class clsAttn_Linear(nn.Module):
    # Inputs:
    #   cls_dim - Dimension of the class embeddings
    #   inCh - Input channels in the input embeddings
    def __init__(self, cls_dim, inCh):
        super(clsAttn_Linear, self).__init__()
        self.inCh = torch.tensor(inCh)
        
        # Query and Key embedding matrices
        self.Q_emb = nn.Linear(cls_dim, inCh)
        self.K_emb = nn.Linear(cls_dim, inCh)

        # Output embedding transformation
        self.out_emb = nn.Sequential(
            nn.Conv2d(inCh, inCh, 1),
            nn.GroupNorm(inCh//4 if inCh > 4 else 1, inCh)
        )
        
        # Layer normalization
        self.LN = nn.GroupNorm(inCh//4 if inCh > 4 else 1, inCh)

        # Attention matrix softmax
        self.softmax = nn.Softmax(-1)
        
        
    # Given an input tensor and class embeddings, attend to the
    # input tensor using the class embeddings
    # Inputs:
    #   X - tensor of shape (N, inCh, L, W) to attend to
    #   cls - tensor of shape (N, cls_emb) to make the attention with
    # Outputs:
    #   Tensor of shape (N, inCh, L, W)
    def forward(self, X, cls):
        res = X.clone()

        # Normalize input
        X = self.LN(X)

        # Get the keys and queries
        # cls: (N, cls_emb) -> (N, inCh)
        K, Q = self.Q_emb(cls), self.K_emb(cls)

        # Extend the class embeddings
        # (N, inCh) -> (N, inCh, 1)
        K = K.unsqueeze(-1)
        Q = Q.unsqueeze(-1)

        # Softmax activation applied to keys and queries
        K = torch.nn.functional.elu(K)+1
        Q = torch.nn.functional.elu(Q)+1

        # Scale the queries
        Q = Q/torch.sqrt(self.inCh)

        # Apply the keys matrix to the input embeddings
        X = torch.einsum("nclw, nco -> nolw", X, K)

        # Apply the queries matrix to the input embeddings
        X = torch.einsum("nolw, nco -> nclw", X, Q)

        # Output embeddings
        X = self.out_emb(X) + res









import math
class Efficient_Cls_Attention(nn.Module):
    # Efficient channel attention based on
    # https://arxiv.org/abs/1910.03151

    # Inputs:
    #   channels - Number of channels in the input
    #   gamma, b - gamma and b parameters of the kernel size calculation
    def __init__(self, cls_dim, channels, gamma=2, b=1):
        super(Efficient_Cls_Attention, self).__init__()

        # Linear layer to convert from cls_dim to C
        self.latent_dim_chng = nn.Linear(cls_dim, channels) if cls_dim != channels else nn.Identity()

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
    #   cls - Input class embeddings of shape (N, cls_dim)
    # Outputs:
    #   Image tensor of shape (N, C, L, W)
    def forward(self, X, cls):
        # Save the input tensor as a residual
        res = X.clone()

        # Apply the class latent dimension change
        cls = self.latent_dim_chng(cls.clone())

        # Reshape the input tensor to be of shape (N, 1, C)
        cls = cls.unsqueeze(-2)

        # Compute the channel attention
        cls = self.conv(cls)

        # Apply the sigmoid function to the channel attention
        cls = self.sigmoid(cls)

        # Reshape the input tensor to be of shape (N, C, 1, 1)
        cls = cls.permute(0, 2, 1).unsqueeze(-1)

        # Scale the input by the attention
        return res * cls