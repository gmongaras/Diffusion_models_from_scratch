import torch
from torch import nn


cpu = torch.device('cpu')
gpu = torch.device('cuda:0')




# Trains a diffusion model
class model_trainer():
    # diff_model - A diffusion model to train
    # batchSize - Batch size to train the model with
    # epochs - Number of epochs to train the model for
    # lr - Learning rate of the model optimizer
    # device - Device to put the model and data on (cpu, gpu, or partgpu)
    def __init__(self, diff_model, batchSize, epochs, lr, device):
        # Saved info
        self.T = diff_model.T
        self.model = diff_model
        self.batchSize = batchSize
        self.epochs = epochs
        
        # Convert the device to a torch device
        if device.lower() == "fullgpu":
            if torch.cuda.is_available():
                dev = device.lower()
                device = torch.device('cuda:0')
            elif torch.has_mps == True:
                dev = "mps"
                device = torch.device('mps')
            else:
                dev = "cpu"
                print("GPU not available, defaulting to CPU. Please ignore this message if you do not wish to use a GPU\n")
                device = torch.device('cpu')
        else:
            dev = device.lower()
            device = torch.device('cpu')
        self.device = device
        self.dev = dev
        
        # Put the model on the desired device
        if self.dev == "partgpu":
            self.model.to(gpu)
        else:
            self.model.to(self.device)
            
        # Uniform distribution for values of t
        self.T_dist = torch.distributions.uniform.Uniform(float(1.0), float(self.T)) 
        
        # Optimizer
        self.optim = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        
    # Norm function over batches
    def norm_2(self, A, B):
        return torch.sqrt((A.flatten(1, -1)**2 + B.flatten(1, -1)**2).sum(1))
    
    # Simple loss function (L_simple)
    def loss_simple(self, epsilon, epsilon_pred):
        return self.norm_2(epsilon, epsilon_pred).mean()
    
    
    # Loss functions
    def lossFunct(self, epsilon,):
        pass
        
    
    
    # Trains the saved model
    # Inputs:
    #   X - A batch of images of shape (B, C, L, W)
    def train(self, X):
        
        for epoch in range(1, self.epochs+1):
            # Get a sample of `batchSize` number of images
            batch = X[torch.randperm(X.shape[0])[:self.batchSize]]
            
            # Get values of t to noise the data
            t_vals = self.T_dist.sample((self.batchSize,))
            
            # Noise the batch of data
            batch_y, epsilon_y = self.model.noise_batch(batch, t_vals)
            
            # Send the noised data through the model to get the
            # predicted noise and variance
            epsilon_x, variance_x = self.model(batch_y)
            
            # Get the loss (for now it's just the noise prediction)
            loss = self.loss_simple(epsilon_y, epsilon_x)
            
            # Optimize the model
            loss.backward()
            self.optim.step()
            self.optim.zero_grad()