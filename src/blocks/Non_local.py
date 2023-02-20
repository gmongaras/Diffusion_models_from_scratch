from torch import nn






class Non_local(nn.Module):
    # Inputs:
    #   inCh - Number of input channels
    def __init__(self, inCh):
        super(Non_local, self).__init__()
        
        # Query, Key, Value convolutions
        self.Q_conv = nn.Conv3d(inCh, inCh//2, 1)
        self.K_conv = nn.Conv3d(inCh, inCh//2, 1)
        self.V_conv = nn.Conv3d(inCh, inCh//2, 1)
        
        # Output convolution
        self.O_conv = nn.Conv3d(inCh//2, inCh, 1)
        
        # Batch normalization
        self.batchNorm = nn.BatchNorm2d(inCh)
        
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
        # X: (N, inCh, T, L, W) -> (N, inCh/2, T, L, W)
        K, Q, V = self.Q_conv(X), self.K_conv(X), self.V_conv(X)
        
        # Combine the temporal, length, and width dimenions
        # (N, inCh/2, T, L, W) -> (N, inCh/2, TLW)
        K = K.flatten(start_dim=2)
        Q = Q.flatten(start_dim=2)
        V = V.flatten(start_dim=2)
        
        # Multiply the query and key along the channels 
        # (N, inCh/2, TLW) * (N, inCh/2, TLW) -> (N, TLW, TLW)
        Out = Q.permute(0, 2, 1)@K
        
        # Multiply the output matrix by the values matrix
        # (N, TLW, TLW) * (N, inCh/2, TLW) -> (N, inCh/2, TLW)
        Out = (Out@V.permute(0, 2, 1)).permute(0, 2, 1)
        
        # Unflatten the resulting tensor
        # (N, inCh/2, TLW) -> (N, inCh/2, T, L, W)
        Out = Out.unflatten(-1, X.shape[2:])
        
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