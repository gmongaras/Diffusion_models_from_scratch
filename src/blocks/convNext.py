from torch import nn














class convNext(nn.Sequential):
    # Inputs:
    #   inCh - Number of channels the input batch has
    #   outCh - Number of chanels the ouput batch should have
    #   t_dim - (optional) Number of dimensions in the time input embedding
    #   dropoutRate - Rate to apply dropout to each layer in the block
    def __init__(self, inCh, outCh, t_dim=None, dropoutRate=0.0):
        super(convNext, self).__init__()

        # Implementation found at https://arxiv.org/pdf/2201.03545.pdf
        # It has the following structure:
        #   7x7 conv
        #   Layer Norm
        #   1x1 conv
        #   GELU
        #   Layer Norm
        #   1x1 conv
        self.block = nn.Sequential(
            nn.Conv2d(inCh, inCh, 7, padding=3, groups=inCh),
            nn.GroupNorm(1, inCh),
            nn.Dropout2d(dropoutRate),
            nn.Conv2d(inCh, inCh*2, 1),
            nn.GELU(),
            # nn.GroupNorm(1, inCh*2),
            nn.Conv2d(inCh*2, outCh, 1),
        )

        # Residual path
        self.res = nn.Conv2d(inCh, outCh, 1) if inCh != outCh else nn.Identity()

        # Optional time vector applied over the channels
        if t_dim != None:
            self.timeProj = nn.Linear(t_dim, inCh)
        else:
            self.timeProj = None

    
    # Input:
    #   X - Tensor of shape (N, inCh, L, W)
    #   t - (optional) vector of shape (N, t_dim)
    # Output:
    #   Tensor of shape (N, outCh, L, W)
    def forward(self, X, t=None):
        # Quick t and c check
        if t != None and self.timeProj == None:
            raise RuntimeError("t_dim cannot be None when using time embeddings")

        # Residual connection
        res = self.res(X)

        # Main section
        if t == None:
            X = self.block(X)
        else:
            # Initial convolution and dropout
            X = self.block[0](X)
            X = self.block[1](X)
            X = self.block[2](X)

            # Time embedding
            t = self.timeProj(t).unsqueeze(-1).unsqueeze(-1)

            # Combine the class, time, and embedding information
            X = X + t

            # Output linear projection
            for b in self.block[3:]:
                X = b(X)

        # Connect the residual and main sections
        return X + res