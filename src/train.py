import pickle
from models.diff_model import diff_model
from model_trainer import model_trainer
import os
import click






def train():
    #### Parameters
    
    ## Model params
    inCh = 3
    embCh = 128
    chMult = 1
    num_blocks = 3
    blk_types = ["res", "conv", "clsAtn", "atn", "chnAtn"]
                        # blk_types - How should the residual block be structured 
                        #             (list of "res", "conv", "clsAtn", "atn', and/or "chnAtn". 
                        #              Ex: ["res", "res", "conv", "clsAtn", "chnAtn"] 
    T = 1000
    Lambda = 0.001
    beta_sched = "cosine"
    batchSize = 12
    numSteps = 1            # Number of steps to breakup the batchSize into. Instead
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
    atn_resolution = 16
    dropoutRate = 0.1
    use_importance = False # Should importance sampling be used to sample values of t?
    data_path = "data/Imagenet64"
    load_into_mem = True   # True to load all data into memory first, False to load from disk as needed
    
    ## Saving params
    saveDir = "models/"
    numSaveSteps = 10000
    
    ## Loading params
    loadModel = False
    loadDir = "models/"
    loadFile = "model_12e_100s.pkl"
    optimFile = "optim_12e_100s.pkl"
    loadDefFile = "model_params_12e_100s.json"
    
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
    model = diff_model(inCh, embCh, chMult, num_blocks, blk_types, T, beta_sched, t_dim, device, c_dim, num_classes, atn_resolution, dropoutRate)
    
    # Optional model loading
    if loadModel == True:
        model.loadModel(loadDir, loadFile, loadDefFile)
    
    # Train the model
    trainer = model_trainer(model, batchSize, numSteps, epochs, lr, device, Lambda, saveDir, numSaveSteps, use_importance, p_uncond, load_into_mem=load_into_mem, optimFile=None if loadModel==False or optimFile==None else loadDir+os.sep+optimFile)
    trainer.train(data_path, num_data, cls_min, reshapeType)
    # trainer.train(img_data, labels if c_dim != None else None)
    
    
    
    
    
if __name__ == '__main__':
    train()