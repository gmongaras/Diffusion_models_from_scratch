import torch
from torch import nn






class Non_local_MH(nn.Module):
    # Inputs:
    #   inCh - Number of input channels
    #   num_heads - Number of heads in the multi-head
    #               non-local block
    #   head_res - Optional parameter. Specify the resolution each
    #              head operates at rather than the number of heads. If
    #              this is not None, num_heads is ignored
    #   spatial - Should spatial or channel attion be used
    def __init__(self, inCh, num_heads=2, head_res=None, spatial=False):
        super(Non_local_MH, self).__init__()
        self.num_heads = num_heads
        
        if head_res == None:
            assert inCh%num_heads == 0, "Number of channels must be divisible by the number of heads"
        else:
            assert inCh%head_res == 0, "Number of channels must be divisible by the head resolution"
        
        # Get the number of heads using the head resolution
        if head_res != None:
            self.num_heads = inCh//head_res
        
        # Query, Key, Value convolutions
        self.Q_conv = nn.Conv2d(inCh, inCh, 1)
        self.K_conv = nn.Conv2d(inCh, inCh, 1)
        self.V_conv = nn.Conv2d(inCh, inCh, 1)
        
        # Output convolution
        self.O_conv = nn.Conv2d(inCh, inCh, 1)
        
        # Layer normalization
        self.LN = nn.GroupNorm(inCh//4 if inCh > 4 else 1, inCh)

        # Is this spatial or channel attention
        self.spatial = spatial
        
    # Given a tensor, the tensor is extended to multiple heads
    # Inputs:
    #   X - tensor of shape (N, inCh, L, W)
    # Outputs:
    #   Tensor of shape (N, H, inCh/H, L, W)
    def add_heads(self, X):
        X_shape = X.shape
        return X.reshape(X_shape[0], self.num_heads, X_shape[1]//self.num_heads, *X_shape[2:])
    
    
    # Given a tensor, the tensor is contracted to remove the heads
    # Inputs:
    #   X - tensor of shape (N, H, inCh/H, L, W)
    # Outputs:
    #   Tensor of shape (N, inCh, L, W)
    def remove_heads(self, X):
        X_shape = X.shape
        return X.reshape(X_shape[0], X_shape[1]*X_shape[2], *X_shape[3:])
        
    # Inputs:
    #   X - tensor of shape (N, inCh, L, W)
    # Outputs:
    #   Tensor of shape (N, inCh, L, W)
    def forward(self, X):
        res = X.clone()

        # Get the key, query, and values
        # X: (N, inCh, L, W) -> (N, inCh, L, W)
        K, Q, V = self.Q_conv(X), self.K_conv(X), self.V_conv(X)
        
        # Add H number of heads to each of the embeddings
        # (N, inCh, L, W) -> (N, H, inCh/H, L, W)
        K = self.add_heads(K)
        Q = self.add_heads(Q)
        V = self.add_heads(V)
        
        # Combine the length and width dimenions
        # (N, H, inCh/H, L, W) -> (N, H, inCh/H, LW)
        K = K.flatten(start_dim=3)
        Q = Q.flatten(start_dim=3)
        V = V.flatten(start_dim=3)
        
        # Channel attention finds relationships among channels
        if self.spatial == False:
            # Multiply the query and key along the channels 
            # (N, H, inCh/H, LW) * (N, H, inCh/H, LW) -> (N, H, LW, LW)
            Out = torch.einsum("nhcd, nhcD -> nhdD", Q, K)
            
            # Multiply the output matrix by the values matrix
            # (N, H, LW, LW) * (N, H, inCh/H, LW) -> (N, H, inCh/H, LW)
            Out = torch.einsum("nhdD, nhcD -> nhcd", Out, V)

        # Spatial attention finds relationships among the length and width
        else:
            # Multiply the query and key along the spatial dimension
            # (N, H, inCh/H, LW) * (N, H, inCh/H, LW) -> (N, H, inCh/H, inCh/H)
            Out = Q@K.permute(0, 1, 3, 2)
            
            # Multiply the output matrix by the values matrix
            # (N, H, LW, LW) * (N, H, inCh/H, LW) -> (N, H, inCh/H, LW)
            Out = Out@V
        
        # Unflatten the resulting tensor
        # (N, H, inCh/H, LW) -> (N, H, inCh/H, L, W)
        Out = Out.unflatten(-1, X.shape[2:])
        
        # Reshape the tensor back to its original shape without heads
        # (N, H, inCh/H, L, W) -> (N, inCh, L, W)
        Out = self.remove_heads(Out)
        
        # Send the resulting tensor through the
        # final convolution to get the initial channels
        Out = self.O_conv(Out)
        
        # Return the output with the input as a residual
        return self.LN(Out) + res