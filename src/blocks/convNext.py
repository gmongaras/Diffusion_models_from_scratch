from torch import nn














class convNext(nn.Sequential):
    # Inputs:
    #   inCh - Number of channels the input batch has
    #   outCh - Number of chanels the ouput batch should have
    #   use_t - Should time conditioning be used?
    #   t_dim - Number of dimensions in the time input embedding
    def __init__(self, inCh, outCh, use_t=False, t_dim=None):
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
            nn.Conv2d(inCh, inCh, 7, padding=3),
            nn.GroupNorm(inCh, inCh),
            nn.Conv2d(inCh, outCh, 1),
            nn.GELU(),
            nn.GroupNorm(outCh, outCh),
            nn.Conv2d(outCh, outCh, 1),
        )

        # Residual path
        self.res = nn.Conv2d(inCh, outCh, 1)

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
            t = self.clsProj(t).unsqueeze(-1).unsqueeze(-1)
            X += t
            for b in self.block[1:]:
                X = b(X)
        else:
            X = self.block(X)
        return X + res