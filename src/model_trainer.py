import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import os

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from CustomDataset import CustomDataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.dataloader import DataLoader

try:
    from helpers.multi_gpu_helpers import is_main_process
except ModuleNotFoundError:
    from .helpers.multi_gpu_helpers import is_main_process


cpu = torch.device('cpu')




def init_distributed():

    # Initializes the distributed backend which will take care of synchronizing nodes/GPUs
    dist_url = "env://" # default

    # only works with torch.distributed.launch // torchrun
    rank = int(os.environ["RANK"])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    
    # Try the nccl backend
    try:
        dist.init_process_group(
                backend="nccl",
                init_method=dist_url,
                world_size=world_size,
                rank=rank)
    # Use the gloo backend if nccl isn't supported
    except RuntimeError:
        dist.init_process_group(
                backend="gloo",
                init_method=dist_url,
                world_size=world_size,
                rank=rank)

    # this will make all .cuda() calls work properly
    torch.cuda.set_device(local_rank)

    # synchronizes all the threads to reach this point before moving on
    dist.barrier()




# Trains a diffusion model
class model_trainer():
    # diff_model - A diffusion model to train
    # batchSize - Batch size to train the model with
    # numSteps - Number of steps to breakup the batchSize into. Instead
    #            of taking 1 massive step where the whole batch is loaded into
    #            memory, the batchSize is broken up into sizes of
    #            batchSize//numSteps so that it can fit into memory. Mathematically,
    #            the update will be the same, as a single batch update, but
    #            the update is distributed across smaller updates to fit into memory.
    # epochs - Number of epochs to train the model for
    # lr - Learning rate of the model optimizer
    # device - Device to put the model and data on (gpu or cpu)
    # saveDir - Directory to save the model to
    # numSaveSteps - Number of steps until saving the models
    # use_importance - True to use importance sampling to sample values of t,
    #                  False to use uniform sampling.
    # p_uncond - Probability of training on a null class (only used if class info is used)
    # load_into_mem - True to load all data into memory first, False to load from disk as needed
    # optimFile - Optional name of optimizer to load in
    def __init__(self, diff_model, batchSize, numSteps, epochs, lr, device, Lambda, saveDir, numSaveSteps, use_importance, p_uncond=None, max_world_size=None, load_into_mem=False, optimFile=None):
        # Saved info
        self.T = diff_model.T
        self.batchSize = batchSize//numSteps
        self.numSteps = numSteps
        self.epochs = epochs
        self.Lambda = Lambda
        self.saveDir = saveDir
        self.numSaveSteps = numSaveSteps
        self.use_importance = use_importance
        self.p_uncond = p_uncond
        self.load_into_mem = load_into_mem
        
        # Convert the device to a torch device
        if device.lower() == "gpu":
            if torch.cuda.is_available():
                dev = device.lower()
                local_rank = int(os.environ['LOCAL_RANK']) if max_world_size is None else min(int(os.environ['LOCAL_RANK']), max_world_size)
                device = torch.device(f"cuda:{local_rank}")
            else:
                dev = "cpu"
                print("GPU not available, defaulting to CPU. Please ignore this message if you do not wish to use a GPU\n")
                device = torch.device('cpu')
        else:
            dev = "cpu"
            device = torch.device('cpu')
        self.device = device
        self.dev = dev
        
        # Put the model on the desired device
        if dev != "cpu":
            # Initialize the environment
            init_distributed()

            self.model = DDP(diff_model.cuda(), device_ids=[local_rank], find_unused_parameters=False)
        else:
            self.model = diff_model.cpu()
        # self.model.to(self.device)
            
        # Uniform distribution for values of t from [1:T]
        self.t_vals = np.arange(1, self.T.detach().cpu().numpy()+1)
        self.T_dist = torch.distributions.uniform.Uniform(float(1)-float(0.499), float(self.T)+float(0.499))
        
        # Optimizer
        self.optim = torch.optim.AdamW(self.model.parameters(), lr=lr, eps=1e-4)

        # Load in optimizer paramters if they exist
        if optimFile:
            self.optim.load_state_dict(torch.load(optimFile, map_location=self.device))
        
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


    # Variational Lower Bound loss function which computes the
    # KL divergence between two gaussians
    # Formula derived from: https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
    # Inputs:
    #   t - Values of t of shape (N)
    #   mean_real - The mean of the real distribution of shape (N, C, L, W)
    #   mean_fake - Mean of the predicted distribution of shape (N, C, L, W)
    #   var_real - Variance of the real distribution of shape (N, C, L, W)
    #   var_fake - Variance of the predicted distribution of shape (N, C, L, W)
    # Outputs:
    #   Loss vector for each part of the entire batch
    def loss_vlb_gauss(self, t, mean_real, mean_fake, var_real, var_fake):
        std_real = torch.sqrt(var_real)
        std_fake = torch.sqrt(var_fake)

        # Note:
        # p (mean_real, std_real) - Distribution we want the model to predict
        # q (mean_fake, std_fake) - Distribution the model is predicting
        output = (torch.log(std_fake/std_real) \
            + ((var_real) + (mean_real-mean_fake)**2)/(2*(var_fake)) \
            - torch.tensor(1/2))\
            .flatten(1,-1).mean(-1)

        return output
    
    
    # Combined loss
    # Inputs:
    #   epsilon - True epsilon values of shape (N, C, L, W)
    #   epsilon_pred - Predicted epsilon values of shape (N, C, L, W)
    #   x_t - The noised image at time t of shape (N, C, L, W)
    #   t - The value timestep of shape (N)
    # Outputs:
    #   Loss as a scalar over the entire batch
    def lossFunct(self, epsilon, epsilon_pred, v, x_0, x_t, t):
        # Put the data on the correct device
        x_0 = x_0.to(epsilon_pred.device)
        x_t = x_t.to(epsilon_pred.device)
        
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
        if self.dev == "cpu":
            mean_t_pred = self.model.noise_to_mean(epsilon_pred, x_t, t, True)
            var_t_pred = self.model.vs_to_variance(v, t)
        else:
            mean_t_pred = self.model.module.noise_to_mean(epsilon_pred, x_t, t, True)
            var_t_pred = self.model.module.vs_to_variance(v, t)


        ### Preparing for the real normal distribution

        # Get the scheduler information
        if self.dev == "cpu":
            beta_t = self.model.scheduler.sample_beta_t(t)
            a_bar_t = self.model.scheduler.sample_a_bar_t(t)
            a_bar_t1 = self.model.scheduler.sample_a_bar_t1(t)
            beta_tilde_t = self.model.scheduler.sample_beta_tilde_t(t)
            sqrt_a_bar_t1 = self.model.scheduler.sample_sqrt_a_bar_t1(t)
            sqrt_a_t = self.model.scheduler.sample_sqrt_a_t(t)
        else:
            beta_t = self.model.module.scheduler.sample_beta_t(t)
            a_bar_t = self.model.module.scheduler.sample_a_bar_t(t)
            a_bar_t1 = self.model.module.scheduler.sample_a_bar_t1(t)
            beta_tilde_t = self.model.module.scheduler.sample_beta_tilde_t(t)
            sqrt_a_bar_t1 = self.model.module.scheduler.sample_sqrt_a_bar_t1(t)
            sqrt_a_t = self.model.module.scheduler.sample_sqrt_a_t(t)

        # Get the true mean distribution
        mean_t = ((sqrt_a_bar_t1*beta_t)/(1-a_bar_t))*x_0 +\
            ((sqrt_a_t*(1-a_bar_t1))/(1-a_bar_t))*x_t
        
        # Get the losses
        loss_simple = self.loss_simple(epsilon, epsilon_pred)
        loss_vlb = self.loss_vlb_gauss(t, mean_t, mean_t_pred.detach(), beta_tilde_t, var_t_pred)*self.Lambda

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
        
    
    
    # Trains the model
    # Inputs:
    #   data_path - Path to the data to load in
    #   num_data - Number of datapoints loaded
    #   cls_min - What is the nim calss value
    #   reshapeType - Determines how data should be reshaped
    def train(self, data_path, num_data, cls_min, reshapeType):

        # Was class information given?
        if self.dev == "cpu":
            if self.model.c_emb is not None:
                useCls = True

                # Class assertion
                assert self.p_uncond != None, "p_uncond cannot be None when using class information"
            else:
                useCls = False
        else:
            if self.model.module.c_emb is not None:
                useCls = True

                # Class assertion
                assert self.p_uncond != None, "p_uncond cannot be None when using class information"
            else:
                useCls = False

        # Put the model is train mode
        self.model.train()

        # Create a sampler and loader over the dataset
        dataset = CustomDataset(data_path, num_data, cls_min, scale=reshapeType, loadMem=self.load_into_mem)
        if self.dev == "cpu":
            data_loader = DataLoader(dataset, batch_size=self.batchSize,
                pin_memory=True, num_workers=0, 
                drop_last=False, shuffle=True
            )
        else:
            data_loader = DataLoader(dataset, batch_size=self.batchSize,
            pin_memory=True, num_workers=0, 
            drop_last=False, sampler=
                DistributedSampler(dataset, shuffle=True)
            )

        # Losses over epochs
        self.losses_comb = np.array([])
        self.losses_mean = np.array([])
        self.losses_var = np.array([])
        self.steps_list = np.array([])

        # Number of steps taken
        num_steps = self.model.module.defaults["step"]

        # Cumulative loss over the batch over each set of steps
        losses_comb_s = torch.tensor(0.0, requires_grad=False)
        losses_mean_s = torch.tensor(0.0, requires_grad=False)
        losses_var_s = torch.tensor(0.0, requires_grad=False)
        
        # Iterate over the desiered number of epochs
        for epoch in range(self.model.module.defaults["epoch"], self.epochs+1):
            # Set the epoch number for the dataloader to seed the
            # randomization of the sampler
            if self.dev != "cpu":
                data_loader.sampler.set_epoch(epoch)

            # Iterate over all data
            for step, data in enumerate(data_loader):
                batch_x_0, batch_class = data
                
                # Increate the number of steps taken
                num_steps += 1
                
                # Get values of t to noise the data
                # Sample using weighted values if each t has 10 loss values
                if self.use_importance == True and np.sum(self.losses_ct) == self.losses.size - 20:
                    # Weights for each value of t
                    p_t = np.sqrt((self.losses**2).mean(-1))
                    p_t = p_t / p_t.sum()

                    # Sample the values of t
                    t_vals = torch.tensor(np.random.choice(self.t_vals, size=batch_x_0.shape[0], p=p_t), device=batch_x_0.device)
                # Sample uniformly until we get to that point or if importance
                # sampling is not used
                else:
                    t_vals = self.T_dist.sample((batch_x_0.shape[0],)).to(self.device)
                    t_vals = torch.round(t_vals).to(torch.long)


                # Probability of class embeddings being the null embedding
                if self.p_uncond != None:
                    probs = torch.rand(batch_x_0.shape[0])
                    nullCls = torch.where(probs < self.p_uncond, 1, 0).to(torch.bool).to(self.device)
                else:
                    nullCls = None
                

                # Noise the batch to time t
                with torch.no_grad():
                    if self.dev == "cpu":
                        batch_x_t, epsilon_t = self.model.noise_batch(batch_x_0, t_vals)
                    else:
                        batch_x_t, epsilon_t = self.model.module.noise_batch(batch_x_0, t_vals)
                
                # Send the noised data through the model to get the
                # predicted noise and variance for batch at t-1
                epsilon_t1_pred, v_t1_pred = self.model(batch_x_t, t_vals, 
                    batch_class if useCls else None, nullCls)

                # Get the loss
                loss, loss_mean, loss_var = self.lossFunct(epsilon_t, epsilon_t1_pred, v_t1_pred, 
                                    batch_x_0, batch_x_t, t_vals)

                # Scale the loss to be consistent with the batch size. If the loss
                # isn't scaled, then the loss will be treated as an independent
                # batch for each step. If it is scaled by the step size, then the loss will
                # be treated as a part of a larger batchsize which is what we want
                # to acheive when using steps.
                loss = loss/self.numSteps
                loss_mean /= self.numSteps
                loss_var /= self.numSteps

                # Backprop the loss, but save the intermediate gradients
                loss.backward()

                # Save the loss values
                losses_comb_s += loss.cpu().detach()
                losses_mean_s += loss_mean.cpu().detach()
                losses_var_s += loss_var.cpu().detach()

                # If the number of steps taken is a multiple of the number
                # of desired steps, update the models
                if num_steps%self.numSteps == 0:
                    # Update the model using all losses over the steps
                    self.optim.step()
                    self.optim.zero_grad()

                    if is_main_process():
                        print(f"step #{num_steps}   Latest loss estimate: {round(losses_comb_s.cpu().detach().item(), 6)}")

                    # Save the loss values
                    self.losses_comb = np.append(self.losses_comb, losses_comb_s.item())
                    self.losses_mean = np.append(self.losses_mean, losses_mean_s.item())
                    self.losses_var = np.append(self.losses_var, losses_var_s.item())
                    self.steps_list = np.append(self.steps_list, num_steps)

                    # Reset the cumulative step loss
                    losses_comb_s *= 0
                    losses_mean_s *= 0
                    losses_var_s *= 0


                # Save the model and graph every number of desired steps
                if num_steps%self.numSaveSteps == 0 and is_main_process():
                    if self.dev == "cpu":
                        self.model.saveModel(self.saveDir, self.optim, epoch, num_steps)
                    else:
                        self.model.module.saveModel(self.saveDir, self.optim, epoch, num_steps)
                    self.graph_losses()

                    print("Saving model")
            
            if is_main_process():
                print(f"Loss at epoch #{epoch}, step #{num_steps}, update #{num_steps/self.numSteps}\n"+\
                        f"Combined: {round(self.losses_comb[-10:].mean(), 4)}    "\
                        f"Mean: {round(self.losses_mean[-10:].mean(), 4)}    "\
                        f"Variance: {round(self.losses_var[-10:].mean(), 6)}\n\n")
    



    # Graph the losses through training
    def graph_losses(self):
        plt.clf()
        fig, ax = plt.subplots()

        ax.set_title("Losses over epochs")
        ax.set_ylabel("Loss")
        ax.set_xlabel("Step")
        # ax.plot(self.epochs_list, self.losses_comb, label="Combined loss")
        ax.plot(self.steps_list, self.losses_mean, label="Mean loss")
        # ax.plot(self.epochs_list, self.losses_var, label="Variance loss")
        ax.legend()
        fig.savefig(self.saveDir + os.sep + "lossGraph.png", format="png")
        plt.close(fig)
