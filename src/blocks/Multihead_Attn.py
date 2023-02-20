import torch
from torch import nn
from torch.nn.functional import interpolate






class Multihead_Attn(nn.Module):
    # Inputs:
    #   inCh - Number of input channels
    #   num_heads - Number of heads in the attention mechanism
    #   resolution - Downsampled resolution of the input image
    #               in the spatial dimension
    #   spatial - Should spatial or channel attion be used
    def __init__(self, inCh, num_heads=2, resolution=16, spatial=False):
        super(Multihead_Attn, self).__init__()
        self.inCh = inCh
        self.num_heads = num_heads
        self.resolution = resolution

        # Downsampler used to downsample the input image
        # when computing the queries and keys to
        # make it more computationally feasable
        self.downsampler = nn.AdaptiveAvgPool2d(resolution)
        
        # Spatial attention performs computation on
        # the channels.
        if spatial == True:
            self.KQV_weight = nn.Conv2d(inCh, inCh*3, 1)
        # Channel attention performs computation
        # in the spatial dimension
        else:
            self.Q_weight = nn.Conv2d(resolution**2, (resolution**2)*3, 1)
        
        # Output convolution
        self.O_conv = nn.Conv2d(inCh, inCh, 1)
        
        # Layer normalization
        self.LN = nn.GroupNorm(inCh//4 if inCh > 4 else 1, inCh)

        # Normalization factor before softmax
        if spatial == True:
            self.norm_factor = 1/torch.sqrt(torch.tensor(inCh))
        else: 
            self.norm_factor = 1/resolution

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
        
    # Inputs:
    #   X - tensor of shape (N, inCh, L, W)
    # Outputs:
    #   Tensor of shape (N, inCh, L, W)
    def forward(self, X):
        # Get the residual
        res = X.clone()

        # Save the original lanegth and width values
        orig_l = X.shape[-2]
        orig_w = X.shape[-1]

        # Normalize the input
        X = self.LN(X)

        # For spatial attention, we first want to
        # get the queries, keys and values, then
        # downsample and flatten
        if self.spatial == True:
            # Get the keys, queries and values
            KQV = self.KQV_weight(X)
            K, Q, V = KQV[:, :self.inCh],\
                KQV[:, self.inCh:self.inCh*2], KQV[:, self.inCh*2:]
            
            # downsample the spatial resolutions in the
            # keys and queries and values
            # K, Q, V: (N, inCh, L, W) -> (N, inCh, resolution, resolution)
            K, Q, V = self.downsampler(K), self.downsampler(Q), self.downsampler(V)

            # Flatten the keys, queries, and values
            # (N, inCh, L, W) -> (N, inCh, LW)
            K = K.flatten(start_dim=-2)
            Q = Q.flatten(start_dim=-2)
            V = V.flatten(start_dim=-2)

        # For channel attention, we first want to
        # downsample then flatten then
        # transpose the channel and spatial
        # dimension, then finally
        # get the keys queries and values
        else:
            # First we downsample
            # X: (N, inCh, L, W) -> (N, inCh, resolution, resolution)
            X = self.downsampler(X)

            # Next X is flattened
            # X: (N, inCh, L, W) -> (N, inCh, LW)
            X = X.flatten(start_dim=-2)

            # Then transpose the channels and
            # spatial dimension
            # X: (N, inCh, LW) -> (N, LW, inCh)
            X = X.transpose(-2, -1)

            # Finally, get the keys, queries, and values
            KQV = self.KQV_weight(X)
            K, Q, V = KQV[:, :self.inCh],\
                KQV[:, self.inCh:self.inCh*2], KQV[:, self.inCh*2:]
        
        # Add H number of heads to each of the embeddings
        # (N, inCh, LW) -> (N, H, inCh/H, LW)
        K = self.add_heads(K)
        Q = self.add_heads(Q)
        V = self.add_heads(V)
        


        # Multiply the query and key along the channels 
        # (N, H, inCh/H, LW) * (N, H, inCh/H, LW) -> (N, H, LW, LW)
        Out = torch.einsum("nhcd, nhcD -> nhdD", Q, K)

        # Normalize
        Out = self.softmax(Out*self.norm_factor)
        
        # Multiply the output matrix by the values matrix
        # (N, H, LW, LW) * (N, H, inCh/H, LW) -> (N, H, inCh/H, LW)
        Out = torch.einsum("nhdD, nhcD -> nhcd", Out, V)

        

        # With the attention computed, we can now resconstruct
        # the image input
        # Channel attention first needs to transpose the
        # channel and spatial dimensions
        if self.spatial == False:
            Out = Out.transpose(-2, -1)
        
        # Unflatten the resulting tensor
        # (N, H, inCh/H, LW) -> (N, H, inCh/H, L, W)
        Out = Out.unflatten(-1, (self.resolution, self.resolution))
        
        # Reshape the tensor back to its original shape without heads
        # (N, H, inCh/H, L, W) -> (N, inCh, L, W)
        Out = self.remove_heads(Out)

        # Upsample the spatial dimenion to its original shape
        Out = interpolate(Out, size=[orig_l, orig_w])
        
        # Send the resulting tensor through the
        # final convolution to get the initial channels
        Out = self.O_conv(Out)
        
        # Return the output with the input as a residual
        return Out + res