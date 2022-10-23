import zipfile
import pickle
import torch
from .models.diff_model import diff_model






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
    num_heads = 8
    num_res_blocks = 2
    T = 500
    beta_sched = "cosine"
    model = diff_model(inCh, embCh, chMult, num_heads, num_res_blocks, T, beta_sched)
    model.noise_batch(torch.rand(32, 3, 16, 16), 100)
    
    
    
    
    
if __name__ == '__main__':
    main()