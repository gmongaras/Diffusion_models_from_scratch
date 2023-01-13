# Path hack for relative paths
import sys, os
sys.path.insert(0, os.path.abspath('./src'))

from src.models.diff_model import diff_model
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch




def test():
    filePath = "./tests/testimg.gif"
    T = 500
    beta_sched = "linear"
    t = 400
    
    # Load in the image as an RGB numpy array
    im = np.array(Image.open(filePath).convert("RGB")).astype(float)
    
    height = im.shape[0]
    width = im.shape[1]
    channels = im.shape[2]
    
    # Create the model
    model = diff_model(channels, 100, 1, 1, 1, T, beta_sched, 100)
    
    # Does unnoising the image work?
    with torch.no_grad():
        un_im = model(torch.tensor(im).to(torch.float32).permute(2, 1, 0).unsqueeze(0))
        
        # Output should two tensors of the same shape
        assert un_im[0].squeeze().shape == (channels, width, height)
        assert un_im[1].squeeze().shape == (channels, width, height)
    
    # What does the noise look like at time t-1 and time t?
    batch_x_t1, epsilon_t1 = model.noise_batch(torch.tensor(im)[0], torch.tensor(t).unsqueeze(0)-1)
    batch_x_t, epsilon_t = model.noise_batch(torch.tensor(im)[0], torch.tensor(t).unsqueeze(0))
    
    # Noise the image and show it
    plt.imshow(model.noise_batch(torch.tensor(im)[0], torch.tensor(t).unsqueeze(0)))
    plt.show()



if __name__ == "__main__":
    test()