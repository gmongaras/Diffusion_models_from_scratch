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
    beta_sched = "cosine"
    t = 400
    
    # Load in the image as an RGB numpy array
    im = np.array(Image.open(filePath).convert("RGB")).astype(float)
    
    height = im.shape[0]
    width = im.shape[1]
    channels = im.shape[2]
    
    # Create the model
    model = diff_model(channels, 100, 1, 1, 1, T, beta_sched)
    
    # Does unnoising the image work?
    with torch.no_grad():
        un_im = model(torch.tensor(im).to(torch.float32).permute(2, 1, 0).unsqueeze(0)).squeeze().permute(1, 2, 0)
        
        # Output should have twice as many channels
        assert un_im.shape == (width, height, channels*2)
    
    # Noise the image and show it
    plt.imshow(model.noise_batch(torch.tensor(im), t))
    plt.show()



if __name__ == "__main__":
    test()