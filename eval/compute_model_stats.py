# Path hack for relative paths above parent
import sys, os
sys.path.insert(0, os.path.abspath('src/'))


import torch
from torch import nn
from torchvision import transforms
import math
import numpy as np

from src.models.diff_model import diff_model

cpu = torch.device("cpu")







# Computes the mean and variance of the given model
# for its FID scores and saves it to a tensor
def compute_model_stats(
        # Load name parameters
        model_dirname="models_res",
        model_filename = "model_152e_190000s.pkl",
        model_params_filename = "model_params_152e_190000s.json",

        # Device to load in
        device = "gpu",
        gpu_num = 0,

        # Batch size and number of images to generate
        num_fake_imgs = 10000,
        batchSize = 200,

        # Generation step size, DDIM scale, correct output?
        step_size = 1,
        DDIM_scale = 1,
        corrected = True,

        # Filenames for outputs
        file_path = "eval/saved_stats/",
        mean_filename = "fake_mean_190K.npy",
        var_filename = "fake_var_190K.npy",
    ):


    # Used to transforms the images to the correct distirbution
    # as shown here: https://pytorch.org/hub/pytorch_vision_inception_v3/
    def normalize(imgs):
        # Convert image to 299x299
        imgs = transforms.Compose([transforms.Resize((299,299))])(imgs)

        # Standardize to [0, 1]
        imgs = imgs/255.0

        # Normalize by mean and std
        return transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])(imgs)






    # Get the device
    if device == "gpu":
        device = torch.device(f"cuda:{gpu_num}")
    else:
        device = torch.device(f"cpu")

    # Load in the model
    model = diff_model(3, 3, 1, 1, ["res", "res"], 100000, "cosine", 100, device, 100, 1000, 16, 0.0, step_size, DDIM_scale)
    model.loadModel(model_dirname, model_filename, model_params_filename)
    model.eval()

    # Load in the inception network
    inceptionV3 = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', weights="Inception_V3_Weights.DEFAULT")
    inceptionV3.eval()
    inceptionV3.to(model.device)

    # Remove the fully connected output layer
    inceptionV3.fc = nn.Identity()
    inceptionV3.aux_logits = False
    


    ### Model Generation ###

    # Generate images and calculate the inceptions scores
    scores = None
    with torch.no_grad():
        for i in range(math.ceil(num_fake_imgs/batchSize)):
            # Get the current batch size
            cur_batch_size = min(num_fake_imgs, batchSize*(i+1))-batchSize*i

            # Generate some images
            imgs = model.sample_imgs(cur_batch_size, use_tqdm=True, unreduce=True, corrected=corrected)

            # Normalize the inputs
            imgs = normalize(imgs.to(torch.uint8))

            # Calculate the inception scores and store them
            if type(scores) == type(None):
                scores = inceptionV3(imgs).to(cpu).numpy().astype(np.float16)
            else:
                scores = np.concatenate((scores, inceptionV3(imgs).to(cpu).numpy().astype(np.float16)), axis=0)

            print(f"Num loaded: {min(num_fake_imgs, batchSize*(i+1))}")
    
    # Delete the model as its no longer needed
    device = model.device
    del model, inceptionV3

    # Calculate the mean and covariance of the generated batch
    mean = np.mean(scores, axis=0)
    var = np.cov(scores, rowvar=False)


    # Save the mean and variance
    np.save(f"{file_path}{os.sep}{mean_filename}", mean)
    np.save(f"{file_path}{os.sep}{var_filename}", var)






if __name__ == "__main__":
    compute_model_stats()