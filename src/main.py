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
    embCh = 32
    chMult = 2
    num_heads = 8
    num_res_blocks = 4
    T = 1000
    Lambda = 0.001
    beta_sched = "cosine"
    batchSize = 128
    device = "gpu"
    epochs = 50000
    lr = 0.0005
    t_dim = 256
    use_importance = False # Should importance sampling be used to sample values of t?
    
    ## Saving params
    saveDir = "models/"
    numSaveEpochs = 1000
    
    ## Loading params
    loadModel = False
    loadDir = "models/"
    loadFile = "model_1000.pkl"
    loadDefFile = "model_params_1000.json"
    
    ## Data parameters
    reshapeType = "down" # Should the data be reshaped to the nearest power of 2 down, up, or not at all?
    
    
    
    
    
    ### Load in the data
    
    # Open the zip file
    archive = zipfile.ZipFile('data/archive.zip', 'r')
    
    # Read the pickle file
    zip_file = archive.open("mini-imagenet-cache-train.pkl", "r")
    zip_file2 = archive.open("mini-imagenet-cache-test.pkl", "r")
    zip_file3 = archive.open("mini-imagenet-cache-val.pkl", "r")
    
    # Load the data
    data = pickle.load(zip_file)
    data2 = pickle.load(zip_file2)
    data3 = pickle.load(zip_file3)
    img_data = data["image_data"]
    class_data = data["class_dict"]
    img_data2 = data2["image_data"]
    class_data2 = data2["class_dict"]
    img_data3 = data2["image_data"]
    class_data3 = data2["class_dict"]
    del data
    del data2
    del data3
    img_data = np.concatenate((img_data, img_data2, img_data3), axis=0)
    img_data = torch.tensor(img_data, dtype=torch.float32, device=torch.device("cpu"))
    img_data = img_data.permute(0, 3, 1, 2)
    del img_data2, img_data3
    # img_data = img_data.reshape([64, 600, 84, 84, 3])




    # from datasets import load_dataset
    # dataset = load_dataset("fashion_mnist")["train"]["image"]
    # img_data = [np.array(i) for i in dataset]
    # img_data = torch.tensor(np.array(img_data)).unsqueeze(-1).permute(0, -1, 1, 2).to(torch.float)
    # inCh = 1
    



    # Reshape the image to the nearest power of 2
    if reshapeType == "down":
        next_power_of_2 = 2**math.floor(math.log2(img_data.shape[-1]))
    elif reshapeType == "up":
        next_power_of_2 = 2**math.ceil(math.log2(img_data.shape[-1]))
    img_data = torch.nn.functional.interpolate(img_data, (next_power_of_2, next_power_of_2))
    
    # Close the archive
    zip_file.close()
    archive.close()
    
    
    
    
    
    ### Model Creation
    model = diff_model(inCh, embCh, chMult, num_heads, num_res_blocks, T, beta_sched, t_dim, device)
    
    # Optional model loading
    if loadModel == True:
        model.loadModel(loadDir, loadFile, loadDefFile,)
    
    # Train the model
    trainer = model_trainer(model, batchSize, epochs, lr, device, Lambda, saveDir, numSaveEpochs, use_importance)
    trainer.train(img_data)
    
    # What does a sample image look like?
    noise = torch.randn((1, *img_data.shape[1:])).to(model.device)
    imgs = []
    for t in tqdm(range(T-1, 1, -1)):
        with torch.no_grad():
            noise = model.unnoise_batch(noise, t)
            imgs.append(torch.clamp(unreduce_image(noise[0]).cpu().detach().int(), 0, 255).permute(1, 2, 0))
            
    # Convert the sample image to 0->255
    # and show it
    noise = torch.clamp(unreduce_image(noise).cpu().detach().int(), 0, 255)
    for img in noise:
        plt.imshow(img.permute(1, 2, 0))
        plt.savefig("fig.png")
        plt.show()

    # Image evolution gif
    fig, ax = plt.subplots()
    for i in range(0, len(imgs)):
        title = plt.text(imgs[i].shape[0]//2, -5, f"t = {i}", ha='center')
        imgs[i] = [plt.imshow(imgs[i], animated=True), title]
    animate = animation.ArtistAnimation(fig, imgs, interval=1, blit=False, repeat_delay=1000)
    animate.save('diffusion.gif', fps=50)
    plt.show()
    
    
    
    
    
if __name__ == '__main__':
    main()