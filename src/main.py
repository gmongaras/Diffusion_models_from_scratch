import zipfile
import pickle
import torch
from .models.diff_model import diff_model
import numpy as np
from PIL import Image
from .model_trainer import model_trainer
import matplotlib.pyplot as plt
from .helpers.image_rescale import reduce_image, unreduce_image






def main():
    #### Parameters
    
    ## Data parameters
    
    
    
    
    
    ### Load in the data
    
    # Open the zip file
    # archive = zipfile.ZipFile('data/archive.zip', 'r')
    
    # # Read the pickle file
    # zip_file = archive.open("mini-imagenet-cache-train.pkl", "r")
    
    # # Load the data
    # data = pickle.load(zip_file)
    # img_data = data["image_data"]
    # class_data = data["class_dict"]
    # del data
    # img_data = img_data.reshape([64, 600, 84, 84, 3])
    
    # # Close the archive
    # zip_file.close()
    # archive.close()
    
    
    
    
    
    ### Model Creation
    inCh = 3
    embCh = 128
    chMult = 2
    num_heads = 2
    num_res_blocks = 1
    T = 400
    Lambda = 0.0001
    beta_sched = "linear"
    batchSize = 2
    device = "gpu"
    epochs = 10
    lr = 0.0001
    t_dim = 100
    model = diff_model(inCh, embCh, chMult, num_heads, num_res_blocks, T, beta_sched, t_dim, device)
    
    # Load in a test image
    filePath = "./tests/testimg.gif"
    im = np.array(Image.open(filePath).convert("RGB")).astype(float)
    im = torch.tensor(im).permute(2, 0, 1).unsqueeze(0).to(torch.float32)
    im = im.repeat(batchSize, 1, 1, 1)
    
    # Train the model
    trainer = model_trainer(model, batchSize, epochs, lr, device, Lambda)
    trainer.train(im)
    
    # What does a sample image look like?
    noise = torch.randn_like(im[:1]).to(model.device)
    for t in range(T, 0, -1):
        with torch.no_grad():
            noise = model.unnoise_batch(noise, t)
            
    # Convert the sample image to 0->255
    # and show it
    noise = unreduce_image(noise).cpu().detach()
    plt.imshow(noise[0].permute(1, 2, 0))
    plt.savefig("fig.png")
    
    
    
    
    
if __name__ == '__main__':
    main()