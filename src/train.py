import pickle
from models.diff_model import diff_model
from model_trainer import model_trainer
import os
import click
from typing import List




def str_to_list(s):
    """
    Convert a string of form 'a,b,c' to a list ['a', 'b', 'c']
    """
    return s.replace(" ", "").split(",")




@click.command()

# Data Parameters
@click.option("--inCh", "inCh", type=int, default=3, help="Number of input channels for the input data", required=False)
@click.option("--data_path", "data_path", type=str, default="data/Imagenet64", help="Path to the ImageNet 64x64 dataset", required=False)
@click.option("--load_into_mem", "load_into_mem", type=bool, default=True, help="True to load all ImageNet data into memory, False to load the data from disk as needed", required=False)

# Model Parameters
@click.option("--embCh", "embCh", type=int, default=128, help="Number of channels in the top layer of the U-net. Note, this is scaled by 2^(chMult*layer) each U-net layer", required=False)
@click.option("--chMult", "chMult", type=int, default=1, help="At each U-net layer, at what scale should the channels be multiplied by? Each layer has embCh*2^(chMult*layer) channels", required=False)
@click.option("--num_layers", "num_blocks", type=int, default=3, help="Number of U-net layers. A value of 3 has a depth of 3, meaning there are 3 down layers and 3 up layers in the U-net", required=False)
@click.option("--blk_types", "blk_types", type=str_to_list, default="res,clsAtn,chnAtn", help="How should the residual block be structured? (list of \"res\", \"conv\", \"clsAtn\", \"atn\", and/or \"chnAtn\". RES: Normal residual block. CONV: ConvNext block. CLSATN: Attention block to include extra class information (Q, K are cls features). ATN: Attention ViT block for the hidden features. CLSATN: Efficient-lightweight attention block over the feature channels. EX: \"res,res,conv,clsAtn,chnAtn\"", required=False)
@click.option("--T", "T", type=int, default=1000, help="Number of timesteps in the diffusion process", required=False)
@click.option("--beta_sched", "beta_sched", type=str, default="cosine", help="Noise scheduler to use in the diffusion process. Can be either \"linear\" or \"cosine\"", required=False)
@click.option("--t_dim", "t_dim", type=int, default=512, help="Dimension of the vector encoding for the time information.", required=False)
@click.option("--c_dim", "c_dim", type=int, default=512, help="Dimension of the vector encoding for the class information. NOTE: Use -1 for no class information", required=False)
@click.option("--atn_resolution", "atn_resolution", type=int, default=16, help="Resolution of the attention block. The resolution splits the image into patches of that resolution to act as vectors. Ex: a resolution of 16 creates 16x16 patches and flattens them as feature vectors.", required=False)

# Training Parameters
@click.option("--Lambda", "Lambda", type=float, default=0.001, help="Weighting term between the variance and mean loss in the model.", required=False)
@click.option("--batchSize", "batchSize", type=int, default=128, help="Batch size on a single GPU. If using multiple GPUs, this batch size will be multiplied by the GPU count.", required=False)
@click.option("--gradAccSteps", "numSteps", type=int, default=1, help="Number of steps to breakup the batchSize into. Instead of taking 1 massive step where the whole batch is loaded into memory, the batchSize is broken up into sizes of batchSize//numSteps so that it can fit into memory. Mathematically, the update will be the same, as a single batch update, but the update is distributed across smaller updates to fit into memory. A higher value takes more time, but uses less memory.", required=False)
@click.option("--device", "device", type=str, default="gpu", help="Device to put the model on. Use \"gpu\" to put it on one or multiple Cuda devices or \"cpu\" to put it on the CPU. CPU should only be used for testing.", required=False)
@click.option("--epochs", "epochs", type=int, default=1000000, help="Number of epochs to train for.", required=False)
@click.option("--lr", "lr", type=float, default=0.0003, help="Model learning rate.", required=False)
@click.option("--p_uncond", "p_uncond", type=int, default=0.2, help="Probability of training on a null class for classifier-free guidance. Note that good values are 0.1 or 0.2. (only used if c_dim is not None)", required=False)
@click.option("--use_importance", "use_importance", type=bool, default=False, help="True to use importance sampling for values of t, False to use uniform sampling.", required=False)

# Saving Parameters
@click.option("--saveDir", "saveDir", type=str, default="models/", help="Directory to save models checkpoints to. NOTE that three files will be saved: the model .pkl file, the model metadata .json file, and the optimizer .pkl file for training reloading", required=False)
@click.option("--numSaveSteps", "numSaveSteps", type=int, default=10000, help="Number of steps until a new model checkpoint is saved. This is not the number of epochs, rather it's the number of time the model has updates. NOTE that three files will be saved: the model .pkl file, the model metadata .json file, and the optimizer .pkl file for training reloading.", required=False)

# Model loading Parameters
@click.option("--loadModel", "loadModel", type=bool, default=False, help="True to load a pretrained model from a checkpoint. False to use a randomly initialized model. Note that all three model files are needed for a successfull restart: the model .pkl file, the model metadata .json file, and the optimizer .pkl file.", required=False)
@click.option("--loadDir", "loadDir", type=str, default="models/", help="Directory of the model files to load in.", required=False)
@click.option("--loadFile", "loadFile", type=str, default="", help="Model .pkl filename to load in. Will looks something like: model_10e_100s.pkl", required=False)
@click.option("--optimFile", "optimFile", type=str, default="", help="Optimizer .pkl filename to load in. Will looks something like: optim_10e_100s.pkl", required=False)
@click.option("--loadDefFile", "loadDefFile", type=str, default="", help="Model metadata .json filename to load in. Will looks something like: model_params_10e_100s.json", required=False)

# Data loading parameters
@click.option("--reshapeType", "reshapeType", type=str, default="", help="If the data is unequal in size, use this to reshape images up by a power of 2, down a power of 2, or not at all (\"up\", \"down\", \"\")", required=False)
def train(
    # Data Params
    inCh: int,
    data_path: str,
    load_into_mem: bool,

    # Model Params
    embCh: int,
    chMult: int,
    num_blocks: int,
    blk_types: List[str],
    T: int,
    beta_sched: str,
    t_dim: int,
    c_dim: int,
    atn_resolution: int,

    # Training Params
    Lambda: float,
    batchSize: int,
    numSteps: int,
    device: str,
    epochs: int,
    lr: float,
    p_uncond: float,
    use_importance: bool,

    # Saving Params
    saveDir: str,
    numSaveSteps: int,

    # Loading params
    loadModel: bool,
    loadDir: str,
    loadFile: str,
    optimFile: str,
    loadDefFile: str,

    # Data Params
    reshapeType: str

    ):

    # Negative class dimension means no class info
    if c_dim == -1:
        c_dim = None

    # I never added dropout to the model :/
    dropoutRate = 0.0

    # Assertions for file loading
    if loadModel == True:
        assert loadFile != "", "A model .pkl filename must be provided when laoding a model checkpoint."
        assert optimFile != "", "A optimizer .pkl filename must be provided when laoding a model checkpoint."
        assert loadDefFile != "", "A model metadata .json filename must be provided when laoding a model checkpoint."
    
    if reshapeType == "":
        reshapeType = None
    
    
    





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
    
    
    
    
    
if __name__ == '__main__':
    train()