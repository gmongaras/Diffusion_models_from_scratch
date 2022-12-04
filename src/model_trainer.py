import torch
from torch import nn
from .helpers.image_rescale import reduce_image, unreduce_image
import numpy as np
import matplotlib.pyplot as plt
import os
import threading


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
    # use_importance - True to use importance sampling to sample values of t,
    #                  False to use uniform sampling.
    def __init__(self, diff_model, batchSize, epochs, lr, device, Lambda, saveDir, numSaveEpochs, use_importance):
        # Saved info
        self.T = diff_model.T
        self.model = diff_model
        self.batchSize = batchSize
        self.epochs = epochs
        self.Lambda = Lambda
        self.saveDir = saveDir
        self.numSaveEpochs = numSaveEpochs
        self.use_importance = use_importance
        
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
        self.t_vals = np.arange(0, self.T.detach().cpu().numpy())
        self.T_dist = torch.distributions.uniform.Uniform(float(0.0)-float(0.499), float(self.T)+float(0.499))
        
        # Optimizer
        self.optim = torch.optim.AdamW(self.model.parameters(), lr=lr)
        
        # Loss function
        self.MSE = nn.MSELoss(reduction="none").to(self.device)



        # Loss cumulator for each value of t
        self.losses = np.zeros((self.T, 10))
        self.losses_ct = np.zeros(self.T, dtype=int)



    # Update the stored loss values for each value of t
    # Inputs:
    #   loss_vec - Vector of shape (batchSize) with the L_vlb loss
    #              for each item in the batch
    #   t - Vector of shape (batchSize) with the t values for each
    #       item in the batch
    def update_losses(self, loss_vec, t):
        # Iterate over all losses and values of t
        for t_val, loss in zip(t, loss_vec):
            # Save the loss value to the losses array
            if self.losses_ct[t_val] == 10:
                self.losses[t_val] = np.concatenate((self.losses[t_val][1:], [loss]))
            else:
                self.losses[t_val, self.losses_ct[t_val]] = loss
                self.losses_ct[t_val] += 1

        
        
    # Simple loss function (L_simple) (MSE Loss)
    # Inputs:
    #   epsilon - True epsilon values of shape (N, C, L, W)
    #   epsilon_pred - Predicted epsilon values of shape (N, C, L, W)
    # Outputs:
    #   Vector loss value for each item in the entire batch
    def loss_simple(self, epsilon, epsilon_pred):
        return ((epsilon_pred - epsilon)**2).flatten(1, -1).mean(-1)


    # KL Divergence loss
    # Inputs:
    #   y_true - Distribution we want the model to predict
    #   y_pred - Predicted distribution the model predicted
    # Outputs:
    #   Vector batch of the KL divergence lossed between the 2 distribution
    def KLDivergence(self, y_true, y_pred):
        # Handling small values
        y_true = torch.where(y_true < 1e-5, y_true+1e-5, y_true)
        y_pred = torch.where(y_pred < 1e-5, y_pred+1e-5, y_pred)
        return (y_true*(y_true.log() - y_pred.log())).flatten(1, -1).mean(-1)
    
    # Variational Lower Bound loss function which computes the
    # KL divergence between two gaussians
    # Formula derived from: https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
    # Inputs:
    #   t - Values of t of shape (N)
    #   mean_real - The mean of the real distribution of shape (N, C, L, W)
    #   mean_fake - Mean of the predicted distribution of shape (N, C, L, W)
    #   var_real - Variance of the real distribution of shape (N, C, L, W)
    #   var_fake - Variance of the predicted distribution of shape (N, C, L, W)
    #   x_0 - The unoised image at time t = 0 of shape (N, C, L, W)
    # Outputs:
    #   Loss vector for each part of the entire batch
    def loss_vlb_gauss(self, t, mean_real, mean_fake, var_real, var_fake, x_0):
        std_real = torch.sqrt(var_real)
        std_fake = torch.sqrt(var_fake)

        # Note:
        # p (mean_real, std_real) - Distribution we want the model to predict
        # q (mean_fake, std_fake) - Distribution the model is predicting
        output = (torch.log(std_fake/std_real) \
            + ((var_real) + (mean_real-mean_fake)**2)/(2*(var_fake)) \
            - torch.tensor(1/2))\
            .flatten(1,-1).mean(-1)
        
        # Replace where t = 0
        output = torch.where(t == 0,
            -torch.distributions.Normal(mean_fake, std_fake).log_prob(x_0).flatten(1,-1).mean(-1),
            output
        )

        return output
    
    
    # Combined loss
    # Inputs:
    #   epsilon - True epsilon values of shape (N, C, L, W)
    #   epsilon_pred - Predicted epsilon values of shape (N, C, L, W)
    #   x_t - The noised image at time t of shape (N, C, L, W)
    #   x_t1 - The unnoised image at time t-1 of shape (N, C, L, W)
    #   t - The value timestep of shape (N)
    # Outputs:
    #   Loss as a scalar over the entire batch
    def lossFunct(self, epsilon, epsilon_pred, v, x_0, x_t, x_t1, t):
        
        """
        There's one important note I looked passed when reading the original
        Denoising Diffusion Probabilistic Models paper. I noticed that L_simple
        was high on low values of t but low on high values of t. I thought
        this was an issue, but it is not. As stated in the paper
        
        "In particular, our diffusion process setup in Section 4 causes the 
        simplified objective to down-weight loss terms corresponding to small t.  
        These terms train the network to denoise data with very small amounts of 
        noise, so it is beneficial to down-weight them so that the network can 
        focus on more difficult denoising tasks at larger t terms"
        (page 5 part 3.4)
        """

        # Get the mean and variance from the model
        mean_t_pred = self.model.noise_to_mean(epsilon_pred, x_t, t, False)
        var_t_pred = self.model.vs_to_variance(v, t)


        ### Preparing for the real normal distribution

        # Get the scheduler information
        beta_t = self.model.scheduler.sample_beta_t(t)
        a_bar_t = self.model.scheduler.sample_a_bar_t(t)
        a_bar_t1 = self.model.scheduler.sample_a_bar_t1(t)
        beta_tilde_t = self.model.scheduler.sample_beta_tilde_t(t)
        sqrt_a_bar_t1 = self.model.scheduler.sample_sqrt_a_bar_t1(t)
        sqrt_a_t = self.model.scheduler.sample_sqrt_a_t(t)

        # Unsqueezing the values to match shape
        beta_t = self.model.unsqueeze(beta_t, -1, 3)
        a_bar_t = self.model.unsqueeze(a_bar_t, -1, 3)
        a_bar_t1 = self.model.unsqueeze(a_bar_t1, -1, 3)
        beta_tilde_t = self.model.unsqueeze(beta_tilde_t, -1, 3)
        sqrt_a_bar_t1 = self.model.unsqueeze(sqrt_a_bar_t1, -1, 3)
        sqrt_a_t = self.model.unsqueeze(sqrt_a_t, -1, 3)

        # Get the true mean distribution
        mean_t = ((sqrt_a_bar_t1*beta_t)/(1-a_bar_t))*x_0 +\
            ((sqrt_a_t*(1-a_bar_t1))/(1-a_bar_t))*x_t
        
        # Get the losses
        loss_simple = self.loss_simple(epsilon, epsilon_pred)
        loss_vlb = self.loss_vlb_gauss(t, mean_t, mean_t_pred.detach(), beta_tilde_t, var_t_pred, x_0)*self.Lambda

        # Get the combined loss
        loss_comb = loss_simple + loss_vlb




        # Update the loss storage for importance sampling
        if self.use_importance:
            with torch.no_grad():
                t = t.detach().cpu().numpy()
                loss = loss_vlb.detach().cpu()
                self.update_losses(loss, t)

                # Have 10 loss values been sampled for each value of t?
                if np.sum(self.losses_ct) == self.losses.size - 20:
                    # The losses are based on the probability for each
                    # value of t
                    p_t = np.sqrt((self.losses**2).mean(-1))
                    p_t = p_t / p_t.sum()
                    loss = loss / torch.tensor(p_t[t], device=loss.device)
                # Otherwise, don't change the loss values



        # Return the losses
        return loss_comb.mean(), loss_simple.mean(), loss_vlb.mean()
        
    
    
    # Trains the saved model
    # Inputs:
    #   X - A batch of images of shape (B, C, L, W)
    def train(self, X):

        # Put the model is train mode
        self.model.train()
        
        # Put the data on the cpu
        X = X.to(cpu)
        
        # # Scale the image to (-1, 1)
        # if X.max() > 1.0:
        #     X = reduce_image(X)

        # Should the images be scaled?
        scaled = True if X.max() > 1.0 else False

        # Losses over epochs
        self.losses_comb = []
        self.losses_mean = []
        self.losses_var = []
        self.epochs_list = []
        
        for epoch in range(1, self.epochs+1):
            # Model saving and graphing
            if epoch%self.numSaveEpochs == 0:
                self.model.saveModel(self.saveDir, epoch)
                self.graph_losses()
            
            # Get a sample of `batchSize` number of images and put
            # it on the correct device
            batch_x_0 = X[torch.randperm(X.shape[0])[:self.batchSize]].to(self.device)

            # Scale the images
            if scaled:
                batch_x_0 = reduce_image(batch_x_0)
            
            # Get values of t to noise the data
            # Sample using weighted values if each t has 10 loss values
            if self.use_importance == True and np.sum(self.losses_ct) == self.losses.size - 20:
                # Weights for each value of t
                p_t = np.sqrt((self.losses**2).mean(-1))
                p_t = p_t / p_t.sum()

                # Sample the values of t
                t_vals = torch.tensor(np.random.choice(self.t_vals, size=self.batchSize, p=p_t), device=batch_x_0.device)
            # Sample uniformly until we get to that point or if importantce
            # sampling is not used
            else:
                t_vals = self.T_dist.sample((self.batchSize,)).to(self.device)
                t_vals = torch.round(t_vals).to(torch.long)
            
            # Noise the batch to time t-1
            batch_x_t1, epsilon_t1 = self.model.noise_batch(batch_x_0, t_vals-1)
            
            # Noise the batch to time t
            batch_x_t, epsilon_t = self.model.noise_batch(batch_x_0, t_vals)
            
            # Send the noised data through the model to get the
            # predicted noise and variance for batch at t-1
            epsilon_t1_pred, v_t1_pred = self.model(batch_x_t, t_vals)
            
            # Get the loss
            loss, loss_mean, loss_var = self.lossFunct(epsilon_t, epsilon_t1_pred, v_t1_pred, 
                                  batch_x_0, batch_x_t, batch_x_t1, t_vals)
            
            # Update the model
            loss.backward()
            self.optim.step()
            self.optim.zero_grad()

            # Save the loss values
            self.losses_comb.append(loss.item())
            self.losses_mean.append(loss_mean.item())
            self.losses_var.append(loss_var.item())
            self.epochs_list.append(epoch)
            
            print(f"Loss at epoch #{epoch}  Combined: {round(loss.item(), 4)}    Mean: {round(loss_mean.item(), 4)}    Variance: {round(loss_var.item(), 6)}")
    



    # Graph losses function for a thread
    def graph_losses_helper(self, fig, ax):
        ax.set_title("Losses over epochs")
        ax.set_ylabel("Loss")
        ax.set_xlabel("Epoch")
        # ax.plot(self.epochs_list, self.losses_comb, label="Combined loss")
        ax.plot(self.epochs_list, self.losses_mean, label="Mean loss")
        # ax.plot(self.epochs_list, self.losses_var, label="Variance loss")
        ax.legend()
        fig.savefig(self.saveDir + os.sep + "lossGraph.png", format="png")
        plt.close(fig)

    # Graph the losses through training
    def graph_losses(self):
        # Async saving
        plt.clf()
        fig, ax = plt.subplots()
        thr = threading.Thread(target=self.graph_losses_helper, args=(fig, ax))
        thr.start()
