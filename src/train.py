import zipfile
import pickle
import torch
from models.diff_model import diff_model
import numpy as np
from model_trainer import model_trainer
import math
import os


def train():
    #### Parameters
    
    ## Model params
    inCh = 3
    embCh = 192
    chMult = 1
    num_res_blocks = 3
    T = 1000
    Lambda = 0.001
    beta_sched = "cosine"
    batchSize = 120
    numSteps = 5            # Number of steps to breakup the batchSize into. Instead
                            # of taking 1 massive step where the whole batch is loaded into
                            # memory, the batchSize is broken up into sizes of
                            # batchSize//numSteps so that it can fit into memory. Mathematically,
                            # the update will be the same, as a single batch update, but
                            # the update is distributed across smaller updates to fit into memory
    device = "gpu"
    epochs = 1000000
    lr = 3e-4
    t_dim = 512
    c_dim = 512             # Embedding dimension for class info (use None to not use class info)
    p_uncond = 0.2          # Probability of training on a null class (only used if c_dim is not None)
    dropoutRate = 0.1
    use_importance = False # Should importance sampling be used to sample values of t?
    data_path = "data/Imagenet64"
    
    ## Saving params
    saveDir = "models/"
    numSaveSteps = 10000
    
    ## Loading params
    loadModel = False
    loadDir = "models/"
    loadFile = "model_10000.pkl"
    loadDefFile = "model_params_10000.json"
    
    ## Data parameters
    reshapeType = None # Should the data be reshaped to the nearest power of 2 "down", "up", or not at all?
    
    
    





    # Load in the metadata
    metadata = pickle.load(open(f"{data_path}{os.sep}metadata.pkl", "rb"))
    cls_min = metadata["cls_min"]
    cls_max = metadata["cls_max"]
    num_data = metadata["num_data"]

    # Get the number of classes
    num_classes = cls_max
    
    
    
    
    
    ### Model Creation
    model = diff_model(inCh, embCh, chMult, num_res_blocks, T, beta_sched, t_dim, device, c_dim, num_classes, dropoutRate)
    
    # Optional model loading
    if loadModel == True:
        model.loadModel(loadDir, loadFile, loadDefFile,)
    
    # Train the model
    trainer = model_trainer(model, batchSize, numSteps, epochs, lr, device, Lambda, saveDir, numSaveSteps, use_importance, p_uncond)
    trainer.train(data_path, num_data, cls_min, reshapeType)
    # trainer.train(img_data, labels if c_dim != None else None)
    
    
    
    
    
if __name__ == '__main__':
    train()