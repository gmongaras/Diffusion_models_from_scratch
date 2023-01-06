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
            nn.GroupNorm(1, inCh)
        )
        
        # Layer normalization
        self.LN = nn.GroupNorm(1, inCh)

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
        X = torch.einsum("nclw, ncC -> nclw", X, KQ)

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
            nn.GroupNorm(1, inCh)
        )
        
        # Layer normalization
        self.LN = nn.GroupNorm(1, inCh)

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
        K = K.softmax(-1)
        Q = Q.softmax(-2)

        # Scale the queries
        Q = Q/torch.sqrt(self.inCh)

        # Apply the keys matrix to the input embeddings
        X = torch.einsum("nclw, nco -> nolw", X, K)

        # Apply the queries matrix to the input embeddings
        X = torch.einsum("nolw, nco -> nclw", X, Q)

        # Output embeddings
        X = self.out_emb(X)
        
        # Return the output with the input as a residual
        return X + res