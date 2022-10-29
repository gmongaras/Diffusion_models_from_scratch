from torch import nn






class Non_local_MH(nn.Module):
    # Inputs:
    #   inCh - Number of input channels
    #   num_heads - Number of heads in the multi-head
    #               non-local block
    #   head_res - Optional parameter. Specify the resolution each
    #              head operates at rather than the number of heads. If
    #              this is not None, num_heads is ignored
    def __init__(self, inCh, num_heads=2, head_res=None):
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
        
        # Batch normalization
        self.batchNorm = nn.BatchNorm2d(inCh)
        
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
        
        # Multiply the query and key along the channels 
        # (N, H, inCh/H, LW) * (N, H, inCh/H, LW) -> (N, H, inCh/H, inCh/H)
        """
        Note, the original Non-local paper multiplies the matrices
        to produce a tensor of shape (N, H, TLW, TLW), but this
        produces massive gradients. Also, the original attention
        paper essentially multiplied so the output was (N, H, S, S)
        which is essentially the same as what I'm doing, (N, H, C, C)
        """
        Out = Q@K.permute(0, 1, 3, 2)
        
        # Multiply the output matrix by the values matrix
        # (N, H, inCh/H, inCh/H) * (N, H, inCh/H, TLW) -> (N, H, inCh/H, TLW)
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
        
        # Return the otuput with the input as a residual
        return self.batchNorm(Out + X)