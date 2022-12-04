from torch import nn














class convNext(nn.Sequential):
    # Inputs:
    #   inCh - Number of channels the input batch has
    #   outCh - Number of chanels the ouput batch should have
    #   use_t - Should time conditioning be used?
    #   t_dim - Number of dimensions in the time input embedding
    #   dropoutRate - Rate to apply dropout to each layer in the block
    def __init__(self, inCh, outCh, use_t=False, t_dim=None, dropoutRate=0.0):
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

        # Optional class vector applied over the channels
        self.use_t = use_t
        if use_t:
            self.clsProj = nn.Linear(t_dim, inCh)

    
    # Input:
    #   X - Tensor of shape (N, inCh, L, W)
    #   t - vector of shape (N, t_dim)
    # Output:
    #   Tensor of shape (N, outCh, L, W)
    def forward(self, X, t=None):
        res = self.res(X)

        if self.use_t and t != None:
            X = self.block[0](X)
            X = self.block[1](X)
            X = self.block[2](X)
            t = self.clsProj(t).unsqueeze(-1).unsqueeze(-1)
            X = X + t
            for b in self.block[3:]:
                X = b(X)
        else:
            X = self.block(X)
        return X + res