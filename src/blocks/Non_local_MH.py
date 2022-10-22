from torch import nn






class Non_local_MH(nn.Module):
    # Inputs:
    #   inCh - Number of input channels
    #   num_heads - Number of heads in the multi-head
    #               non-local block
    def __init__(self, inCh, num_heads):
        super(Non_local_MH, self).__init__()
        self.num_heads = num_heads
        
        assert inCh%num_heads == 0, "Number of channels must be divisible by the number of heads"
        
        # Query, Key, Value convolutions
        self.Q_conv = nn.Conv3d(inCh, inCh, 1)
        self.K_conv = nn.Conv3d(inCh, inCh, 1)
        self.V_conv = nn.Conv3d(inCh, inCh, 1)
        
        # Output convolution
        self.O_conv = nn.Conv3d(inCh, inCh, 1)
        
        # Batch normalization
        self.batchNorm = nn.BatchNorm2d(inCh)
        
    # Given a tensor, the tensor is extended to multiple heads
    # Inputs:
    #   X - tensor of shape (N, inCh, T, L, W)
    # Outputs:
    #   Tensor of shape (N, H, inCh/H, T, L, W)
    def add_heads(self, X):
        X_shape = X.shape
        return X.reshape(X_shape[0], self.num_heads, X_shape[1]//self.num_heads, *X_shape[2:])
    
    
    # Given a tensor, the tensor is contracted to remove the heads
    # Inputs:
    #   X - tensor of shape (N, H, inCh/H, T, L, W)
    # Outputs:
    #   Tensor of shape (N, inCh, T, L, W)
    def remove_heads(self, X):
        X_shape = X.shape
        return X.reshape(X_shape[0], X_shape[1]*X_shape[2], *X_shape[3:])
        
    # Inputs:
    #   X - tensor of shape (N, inCh, L, W) or tensor of shape (N, inCh, T, L, W)
    # Outputs:
    #   Tensor of shape (N, inCh, L, W) or (N, inCh, T, L, W)
    def forward(self, X):
        # Does the input have a temporal dimension?
        hasTemporal = False
        if len(X.shape) > 4:
            hasTemporal = True
            
        # If the input is not temporal, make it temporal
        # (N, inCh, L, W) -> (N, inCh, T, L, W)
        if hasTemporal == False:
            X = X.unsqueeze(2)
        
        # Get the key, query, and values
        # X: (N, inCh, T, L, W) -> (N, inCh, T, L, W)
        K, Q, V = self.Q_conv(X), self.K_conv(X), self.V_conv(X)
        
        # Add H number of heads to each of the embeddings
        # (N, inCh, T, L, W) -> (N, H, inCh/H, T, L, W)
        K = self.add_heads(K)
        Q = self.add_heads(Q)
        V = self.add_heads(V)
        
        # Combine the temporal, length, and width dimenions
        # (N, H, inCh/H, T, L, W) -> (N, H, inCh/H, TLW)
        K = K.flatten(start_dim=3)
        Q = Q.flatten(start_dim=3)
        V = V.flatten(start_dim=3)
        
        # Multiply the query and key along the channels 
        # (N, H, inCh/H, TLW) * (N, H, inCh/H, TLW) -> (N, H, TLW, TLW)
        Out = Q.permute(0, 1, 3, 2)@K
        
        # Multiply the output matrix by the values matrix
        # (N, H, TLW, TLW) * (N, H, inCh/2H, TLW) -> (N, H, inCh/2H, TLW)
        Out = (Out@V.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        
        # Unflatten the resulting tensor
        # (N, H, inCh/H, TLW) -> (N, H, inCh/H, T, L, W)
        Out = Out.unflatten(-1, X.shape[2:])
        
        # Reshape the tensor back to its original shape without heads
        # (N, H, inCh/H, T, L, W) -> (N, inCh, T, L, W)
        Out = self.remove_heads(Out)
        
        # Send the resulting tensor through the
        # final convolution to get the initial channels
        Out = self.O_conv(Out)
        
        # Remove the temporal dimension if the temporal dimension
        # didn't exist in the input
        if not hasTemporal:
            Out = Out.squeeze(2)
            X = X.squeeze(2)
        
        # Return the otuput with the input as a residual
        return self.batchNorm(Out + X)