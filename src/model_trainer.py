import torch
from torch import nn
from .helpers.image_rescale import reduce_image, unreduce_image


cpu = torch.device('cpu')
gpu = torch.device('cuda:0')




# Trains a diffusion model
class model_trainer():
    # diff_model - A diffusion model to train
    # batchSize - Batch size to train the model with
    # epochs - Number of epochs to train the model for
    # lr - Learning rate of the model optimizer
    # device - Device to put the model and data on (gpu or cpu)
    # saveDir - Directory to save the model to
    # numSaveEpochs - Number of epochs until saving the models
    def __init__(self, diff_model, batchSize, epochs, lr, device, Lambda, saveDir, numSaveEpochs):
        # Saved info
        self.T = diff_model.T
        self.model = diff_model
        self.batchSize = batchSize
        self.epochs = epochs
        self.Lambda = Lambda
        self.saveDir = saveDir
        self.numSaveEpochs = numSaveEpochs
        
        # Convert the device to a torch device
        if device.lower() == "gpu":
            if torch.cuda.is_available():
                dev = device.lower()
                device = torch.device('cuda:0')
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
        self.model.to(self.device)
            
        # Uniform distribution for values of t
        self.T_dist = torch.distributions.uniform.Uniform(float(0.0), float(self.T-1))
        
        # Optimizer
        self.optim = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        # Loss function
        self.KL = nn.KLDivLoss(reduction="none").to(device)
        
        
    # Simple loss function (L_simple)
    # Inputs:
    #   epsilon - True epsilon values of shape (N, C, L, W)
    #   epsilon_pred - Predicted epsilon values of shape (N, C, L, W)
    # Outputs:
    #   Scalar loss value over the entire batch
    def loss_simple(self, epsilon, epsilon_pred):
        return torch.nn.functional.mse_loss(epsilon_pred, epsilon)
    
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
    
    
    
    # Loss for the variance
    def loss_variance(self, v, t):
        # Get the beta values for this batch of ts
        beta_t, _, a_bar_t = self.model.get_scheduler_info(t)
        
        # Beta values for the previous value of t
        _, _, a_bar_t1 = self.model.get_scheduler_info(t-1)
        
        # Get the beta tilde value
        beta_tilde_t = ((1-a_bar_t1)/(1-a_bar_t))*beta_t
        beta_tilde_t = self.model.unsqueeze(beta_tilde_t, -1, 3)
        
        # KL loss between the v values and t values
        # Depending on the value of t, get the loss
        loss = torch.where(t==0,
                    -torch.log(v).flatten(1,-1).sum(-1).mean(),
                    self.KL(v, beta_tilde_t).flatten(1,-1).sum(-1).mean()
        ).mean()
        
        return loss
    
    
    # Combined loss
    # Inputs:
    #   epsilon - True epsilon values of shape (N, C, L, W)
    #   epsilon_pred - Predicted epsilon values of shape (N, C, L, W)
    #   x_t - The noised image at time t of shape (N, C, L, W)
    #   x_t1 - The unnoised image at time t-1 of shape (N, C, L, W)
    #   t - The value timestep of shape (N)
    # Outputs:
    #   Loss as a scalar over the entire batch
    def lossFunct(self, epsilon, epsilon_pred, v, x_t, x_t1, t):
        # Get the mean and variance from the model
        mean_t = self.model.noise_to_mean(epsilon_pred, x_t, t)
        var_t = self.model.vs_to_variance(v, t)
        
        """
        Note: The paper states that the loss for the
        variance should be L_vlb, but this is not what they
        use in their implementation. Instead, they use
        the KL divergence between the predictions and
        the Beta_t_tilde values
        """
        
        # Get the losses
        loss_simple = self.loss_simple(epsilon, epsilon_pred)
        # loss_vlb = self.loss_vlb(x_t, x_t1, mean_t, var_t, t)
        loss_var = self.loss_variance(var_t, t)
        
        # Return the combined loss
        return loss_simple + self.Lambda*loss_var
        
    
    
    # Trains the saved model
    # Inputs:
    #   X - A batch of images of shape (B, C, L, W)
    def train(self, X):
        
        # Put the data on the cpu
        X = X.to(cpu)
        
        # Scale the image to (-1, 1)
        if X.max() > 1.0:
            X = reduce_image(X)
        
        for epoch in range(1, self.epochs+1):
            # Model saving
            if epoch%self.numSaveEpochs == 0:
                self.model.saveModel(self.saveDir, epoch)
            
            # Get a sample of `batchSize` number of images and put
            # it on the correct device
            batch_x_0 = X[torch.randperm(X.shape[0])[:self.batchSize]].to(self.device)
            
            # Get values of t to noise the data
            t_vals = self.T_dist.sample((self.batchSize,)).to(self.device)
            t_vals = torch.round(t_vals).to(torch.long)
            
            # Noise the batch to time t-1
            #batch_x_t1, epsilon_t1 = self.model.noise_batch(batch_x_0, t_vals-1)
            
            # Noise the batch to time t
            batch_x_t, epsilon_t = self.model.noise_batch(batch_x_0, t_vals)

            # Get the epsilon value between t and t-1
            #epsilon_real = epsilon_t-epsilon_t1
            
            # Send the noised data through the model to get the
            # predicted noise for batch at t-1
            epsilon_t1_pred = self.model(batch_x_t, t_vals)
            
            # Get the loss
            # loss = self.lossFunct(epsilon_real, epsilon_t1_pred, v_t1_pred, 
            #                       batch_x_t, batch_x_t1, t_vals)
            loss = self.loss_simple(epsilon_t, epsilon_t1_pred)
            
            # Optimize the model
            loss.backward()
            self.optim.step()
            self.optim.zero_grad()
            
            print(f"Loss at epoch #{epoch}: {loss.item()}")