import zipfile
import pickle
import torch
from .models.diff_model import diff_model
import numpy as np
from PIL import Image
from .model_trainer import model_trainer
import matplotlib.pyplot as plt
from .helpers.image_rescale import reduce_image, unreduce_image
import math
from tqdm import tqdm
import matplotlib.animation as animation


def main():
    #### Parameters
    
    ## Model params
    inCh = 3
    embCh = 128
    chMult = 1
    num_res_blocks = 4
    T = 4000
    Lambda = 0.001
    beta_sched = "cosine"
    batchSize = 100
    numSteps = 3            # Number of steps to breakup the batchSize into. Instead
                            # of taking 1 massive step where the whole batch is loaded into
                            # memory, the batchSize is broken up into sizes of
                            # batchSize//numSteps so that it can fit into memory. Mathematically,
                            # the update will be the same, as a single batch update, but
                            # the update is distributed across smaller updates to fit into memory
    device = "gpu"
    epochs = 1000000
    lr = 0.0002
    t_dim = 512
    c_dim = 512            # Embedding dimension for class info (use None to not use class info)
    dropoutRate = 0.1
    use_importance = False # Should importance sampling be used to sample values of t?

    training = True

    ## Generation paramters (only in effect when generating samples, not during training)
    step_size = 100                # Step size to take when generating images
    DDIM_scale = 0          # Scale to transition between a DDIM, DDPM, or in between.
                            # use 0 for pure DDIM and 1 for pure DDPM
    
    ## Saving params
    saveDir = "models/"
    numSaveEpochs = 10000
    
    ## Loading params
    loadModel = False
    loadDir = "models/"
    loadFile = "model_10000.pkl"
    loadDefFile = "model_params_10000.json"
    
    ## Data parameters
    reshapeType = "down" # Should the data be reshaped to the nearest power of 2 down, up, or not at all?
    
    
    
    
    
    ### Load in the data
    
    if training:
        # Open the zip files
        archive1 = zipfile.ZipFile('data/Imagenet64_train_part1.zip', 'r')
        archive2 = zipfile.ZipFile('data/Imagenet64_train_part2.zip', 'r')
        
        # Read the pickle data
        data = []
        labels = []
        for filename in archive1.filelist:
            file = pickle.load(archive1.open(filename.filename, "r"))
            data.append(file["data"])
            labels.append(file["labels"])
            del file
            break
        # for filename in archive2.filelist:
        #     file = pickle.load(archive2.open(filename.filename, "r"))
        #     data.append(file["data"])
        #     labels.append(file["labels"])
        #     del file
        
        # Load the data
        archive1.close()
        archive2.close()
        img_data = np.concatenate((data), axis=0)
        labels = np.concatenate((labels), axis=0)
        del data

        # Convert the data to a tensor
        img_data = torch.tensor(img_data, dtype=torch.float32, device=torch.device("cpu"))
        img_data = img_data.reshape(img_data.shape[0], 3, 64, 64)
    



        # Reshape the image to the nearest power of 2
        if reshapeType == "down":
            next_power_of_2 = 2**math.floor(math.log2(img_data.shape[-1]))
        elif reshapeType == "up":
            next_power_of_2 = 2**math.ceil(math.log2(img_data.shape[-1]))
        if next_power_of_2 != img_data.shape[-1]:
            img_data = torch.nn.functional.interpolate(img_data, (next_power_of_2, next_power_of_2))
    
    # How many classes are there?
    num_classes = None
    if c_dim != None:
        num_classes = labels.max()
        if labels.min() == 1:
            num_classes+=1
    
    
    
    
    
    ### Model Creation
    model = diff_model(inCh, embCh, chMult, num_res_blocks, T, beta_sched, t_dim, device, c_dim, num_classes, dropoutRate, step_size if not training else 1, DDIM_scale)
    
    # Optional model loading
    if loadModel == True:
        model.loadModel(loadDir, loadFile, loadDefFile,)
    
    # Train the model
    if training:
        trainer = model_trainer(model, batchSize, numSteps, epochs, lr, device, Lambda, saveDir, numSaveEpochs, use_importance)
        trainer.train(img_data, labels if c_dim != None else None)
    
    # What does a sample image look like?
    noise, imgs = model.sample_imgs(1, True, True, True)
            
    # Convert the sample image to 0->255
    # and show it
    plt.close('all')
    plt.axis('off')
    noise = torch.clamp(noise.cpu().detach().int(), 0, 255)
    for img in noise:
        plt.imshow(img.permute(1, 2, 0))
        plt.savefig("fig.png", bbox_inches='tight', pad_inches=0, )
        plt.show()

    # Image evolution gif
    plt.close('all')
    fig, ax = plt.subplots()
    ax.set_axis_off()
    for i in range(0, len(imgs)):
        title = plt.text(imgs[i].shape[0]//2, -5, f"t = {i}", ha='center')
        imgs[i] = [plt.imshow(imgs[i], animated=True), title]
    animate = animation.ArtistAnimation(fig, imgs, interval=1, blit=True, repeat_delay=1000)
    animate.save('diffusion.gif', writer=animation.PillowWriter(fps=50))
    # plt.show()
    
    
    
    
    
if __name__ == '__main__':
    main()