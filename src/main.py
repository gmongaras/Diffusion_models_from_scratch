import zipfile
import pickle
import torch
from .models.diff_model import diff_model
import numpy as np
from PIL import Image
from .model_trainer import model_trainer






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
    T = 500
    Lambda = 0.0001
    beta_sched = "cosine"
    batchSize = 2
    device = "cpu"
    epochs = 10
    lr = 0.0001
    model = diff_model(inCh, embCh, chMult, num_heads, num_res_blocks, T, beta_sched)
    
    # Load in a test image
    filePath = "./tests/testimg.gif"
    im = np.array(Image.open(filePath).convert("RGB")).astype(float)
    im = torch.tensor(im).permute(2, 0, 1).unsqueeze(0).to(torch.float32)
    im = im.repeat(2, 1, 1, 1)
    
    # Train the model
    trainer = model_trainer(model, batchSize, epochs, lr, device, Lambda)
    trainer.train(im)
    
    
    
    
    
if __name__ == '__main__':
    main()