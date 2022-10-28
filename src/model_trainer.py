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
    def __init__(self, diff_model, batchSize, epochs, lr, device, Lambda):
        # Saved info
        self.T = diff_model.T
        self.model = diff_model
        self.batchSize = batchSize
        self.epochs = epochs
        self.Lambda = Lambda
        
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
        
        # Loss function
        self.KL = nn.KLDivLoss(reduction="none")
        
        
    # Norm function over batches
    def norm_2(self, A, B):
        return torch.sqrt((A.flatten(1, -1)**2 + B.flatten(1, -1)**2).sum(1))
    
    # Simple loss function (L_simple)
    # Inputs:
    #   epsilon - True epsilon values of shape (N, C, L, W)
    #   epsilon_pred - Predicted epsilon values of shape (N, C, L, W)
    def loss_simple(self, epsilon, epsilon_pred):
        return self.norm_2(epsilon, epsilon_pred).mean()
    
    # Variational Lower Bound loss function
    # Inputs:
    #   x_t - The noised image at time t of shape (N, C, L, W)
    #   x_t1 - The unnoised image at time t-1 of shape (N, C, L, W)
    #   mean_t - Predicted mean at time t of shape (N, C, L, W)
    #   var_t - Predicted variance at time t of shape (N, C, L, W)
    #   t - The value timestep of shape (N)
    # Outputs:
    #   Loss scalar over the entire batch
    def loss_vlb(self, x_t, x_t1, mean_t, var_t, t):
        # Using the mean and variance, send the noised image
        # at time x_t through the distribution with the
        # given mean and variance
        x_t1_pred = self.model.normal_dist(x_t, mean_t, var_t)
        x_t1_pred += 1e-5 # Residual for small probabilities
        
        # Convert the x_t-1 values to p and q for easier notation
        p = x_t1_pred # Predictions
        q = x_t1      # Target
        
        # Depending on the value of t, get the loss
        loss = torch.where(t==0,
                    -torch.log(p).flatten(1,-1).sum(-1).mean(),
                    self.KL(p, q).flatten(1,-1).sum(-1).mean()
        ).mean()
            
        return loss
    
    
    # Loss functions
    # Inputs:
    #   epsilon  
    def lossFunct(self, epsilon, epsilon_pred, v, x_t, x_t1, t):
        # Get the mean and variance from the model
        mean_t = self.model.noise_to_mean(epsilon_pred, x_t, t)
        var_t = self.model.vs_to_variance(v, t)
        
        # Get the losses
        loss_simple = self.loss_simple(epsilon, epsilon_pred)
        loss_vlb = self.loss_vlb(x_t, x_t1, mean_t, var_t, t)
        
        # Return the combined loss
        return loss_simple + self.Lambda*loss_vlb
        
    
    
    # Trains the saved model
    # Inputs:
    #   X - A batch of images of shape (B, C, L, W)
    def train(self, X):
        
        for epoch in range(1, self.epochs+1):
            # Get a sample of `batchSize` number of images
            batch_x_0 = X[torch.randperm(X.shape[0])[:self.batchSize]]
            
            # Get values of t to noise the data
            t_vals = self.T_dist.sample((self.batchSize,))
            
            # Noise the batch to time t-1
            batch_x_t1, epsilon_t1 = self.model.noise_batch(batch_x_0, t_vals-1)
            
            # Noise the batch to time t
            batch_x_t, epsilon_t = self.model.noise_batch(batch_x_0, t_vals)

            # Get the epsilon value between t and t-1
            epsilon_real = epsilon_t-epsilon_t1
            
            # Send the noised data through the model to get the
            # predicted noise and variance for batch at t-1
            epsilon_t1_pred, v_t1_pred = self.model(batch_x_t)
            
            # Get the loss
            loss = self.lossFunct(epsilon_real, epsilon_t1_pred, v_t1_pred, 
                                  batch_x_t, batch_x_t1, t_vals)
            
            # Optimize the model
            loss.backward()
            self.optim.step()
            self.optim.zero_grad()
            
            print(f"Loss at epoch #{epoch}: {loss.item()}")