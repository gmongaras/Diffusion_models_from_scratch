import torch
from torch import nn






class Multihead_Attn(nn.Module):
    # Inputs:
    #   inCh - Number of input channels
    #   num_heads - Number of heads in the attention mechanism
    #   resolution - Downsampled resolution of the input image
    #               in the spatial dimension
    #   spatial - Should spatial or channel attion be used. Spatial attention
    #             finds relationships between spatial tokens in the QK matrix,
    #             channel attention find relatipships between the channels
    def __init__(self, inCh, num_heads=2, resolution=16, spatial=False):
        super(Multihead_Attn, self).__init__()
        self.inCh = inCh
        self.num_heads = num_heads
        self.resolution = resolution
        
        # Used to get all the queries, keys, and values
        self.KQV_weight = nn.Conv2d(inCh, inCh*3, 1)
        
        # Output convolution
        self.O_conv = nn.Conv2d(inCh, inCh, 1)
        
        # Layer normalization
        self.LN = nn.GroupNorm(inCh//4 if inCh > 4 else 1, inCh)

        # Normalization factor before softmax
        if spatial == False:
            self.norm_factor = 1/torch.sqrt(torch.tensor(inCh))
        else: 
            self.norm_factor = 1/torch.sqrt(torch.tensor(resolution))

        # Is this spatial or channel attention
        self.spatial = spatial

        self.softmax = nn.Softmax(-1)
        
    # Given a tensor, the tensor is extended to multiple heads
    # Inputs:
    #   X - tensor of shape (N, inCh, LW)
    # Outputs:
    #   Tensor of shape (N, H, inCh/H, LW)
    def add_heads(self, X):
        X_shape = X.shape
        return X.reshape(X_shape[0], self.num_heads, X_shape[1]//self.num_heads, *X_shape[2:])
    
    
    # Given a tensor, the tensor is contracted to remove the heads
    # Inputs:
    #   X - tensor of shape (N, H, inCh/H, LW)
    # Outputs:
    #   Tensor of shape (N, inCh, LW)
    def remove_heads(self, X):
        X_shape = X.shape
        return X.reshape(X_shape[0], X_shape[1]*X_shape[2], *X_shape[3:])
    


    # Given a tensor, split it into patches
    # Inputs:
    #   X - Tensor of shape (N, inCh, L, W)
    # Outputs:
    #   Tensor of shape (N, (L*W/res**2), inCh, res, res)
    def create_patches(self, X):
        # If the resolution is larger than the image size,
        # return the image itself
        if self.resolution > X.shape[-1]:
            return X.unsqueeze(1)

        res = self.resolution
        return X.unfold(2, res, res).unfold(3, res, res).\
            reshape(X.shape[0], -1, X.shape[1], res, res)
    

    # Given a tensor with patches, remove the patches
    # Inputs:
    #   X - Tensor of shape (N, (LW/res**2), inCh, L/res, W/res)
    # Outputs:
    #   Tensor of shape (N, inCh, L, W)
    def remove_patches(self, X, L, W):
        # If there is only one head, just squeeze the image
        if X.shape[1] == 1:
            return X.squeeze()

        return X.reshape(X.shape[0], X.shape[2], L, W)

        
    # Inputs:
    #   X - tensor of shape (N, inCh, L, W)
    # Outputs:
    #   Tensor of shape (N, inCh, L, W)
    def forward(self, X):
        # Saved input dims
        L = X.shape[-2]
        W = X.shape[-1]

        # Get the residual
        res = X.clone()

        # Normalize the input
        X = self.LN(X)

        # Get the keys, queries and values
        KQV = self.KQV_weight(X)
        K, Q, V = KQV[:, :self.inCh],\
            KQV[:, self.inCh:self.inCh*2], KQV[:, self.inCh*2:]
        
        # Add heads by splitting the image into patches
        # (N, inCh, L, W) -> (N, (LW/res**2), inCh, L/res, W/res)
        K = self.create_patches(K)
        Q = self.create_patches(Q)
        V = self.create_patches(V)

        # Flatten the keys, queries, and values
        # (N, H, inCh, L/res, W/res) -> (N, H, inCh, LW)
        K = K.flatten(start_dim=-2)
        Q = Q.flatten(start_dim=-2)
        V = V.flatten(start_dim=-2)
        


        # Multiply the queries and keys
        # if spatial:
        #   (N, H, inCh, LW) * (N, H, inCh, LW) -> (N, H, LW, LW)
        # if channel:
        #   (N, H, inCh, LW) * (N, H, inCh, LW) -> (N, H, inCh, inCh)
        if self.spatial == True:
            Out = torch.einsum("nhcd, nhce -> nhde", Q, K)
        else:
            Out = torch.einsum("nhcd, nhed -> nhce", Q, K)

        # Normalize
        Out = self.softmax(Out*self.norm_factor)
        
        # Multiply the output matrix by the values matrix
        # if spatial:
        #   (N, H, LW, LW) * (N, H, inCh, LW) -> (N, H, inCh, LW)
        # if channel:
        #   (N, H, inCh, LW) * (N, H, inCh, LW) -> (N, H, inCh, LW)
        if self.spatial == True:
            Out = torch.einsum("nhde, nhce -> nhcd", Out, V)
        else:
            Out = torch.einsum("nhce, nhfd -> nhcd", Out, V)
        
        # Unflatten the resulting tensor
        # (N, H, inCh/H, LW) -> (N, H, inCh/H, L, W)
        if Out.shape[1] > 1:
            Out = Out.unflatten(-1, (self.resolution, self.resolution))
        else:
            Out = Out.unflatten(-1, (L, W))
        
        # Reshape the tensor back to its original shape without heads
        # (N, (LW/res**2), inCh, L/res, W/res) -> (N, inCh, L, W)
        Out = self.remove_patches(Out, L, W)
        
        # Send the resulting tensor through the
        # final convolution to get the initial channels
        # and to find features between heads
        Out = self.O_conv(Out)
        
        # Return the output with the input as a residual
        return Out + res