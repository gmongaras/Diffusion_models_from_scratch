# Path hack for relative paths
import sys, os
sys.path.insert(0, os.path.abspath('./src'))


from src.models.PixelCNN import PixelCNN
import torch
from torchvision import datasets as dts
import matplotlib.pyplot as plt







def PixelCNN_test():
    # Params
    test_size_per = 0.1
    epochs = 20
    batch_size = 128
    
    # Get the MNIST data for testing
    mnisttrainset = dts.MNIST(root='./data', train=True, download=True)
    mnisttestset = dts.MNIST(root='./data', train=False, download=True)
    data = torch.cat((mnisttrainset.data, 
                      mnisttestset.data)).unsqueeze(1).float()
    # Data is of shape (N=70,000, C=1, L=28, W=28)
    
    # Create the model
    model = PixelCNN()
    
    # Shuffle the data
    data = data[torch.randperm(data.shape[0])]
    
    # Get the test and train data
    idx = int(test_size_per*data.shape[0])
    data_test = data[:idx]
    data_train = data[idx:]
    del data
    
    
    
    
    ### Train the model
    for epoch in range(1, epochs+1):
        # Split the data into batches
        batches = torch.split(data_train, batch_size)
        
        # Iterate over all batches
        for batch in batches:
            # Feed the batch through the model
            # Output: (N, 28, 28, 2)
            Y_hat = model(batch).permute(0, 2, 3, 1)
            
            # One-hot encode the labels
            batch = batch.permute(0, 2, 3, 1).squeeze()
            batch = torch.nn.functional.one_hot(batch.long(), Y_hat.shape[-1]).float()
            
            # Get the loss
            loss = model.loss_funct(Y_hat, batch)
            
            # Optimize the model
            loss.backward()
            model.optim.step()
            model.optim.zero_grad()
            
        print(f"epoch {epoch}: {loss.detach().item()}")
            
    
    
    ### Model prediction
    
    # Tensors for the images
    rows = 28
    cols = 28
    channels = 1
    out_img = torch.zeros((1, channels, rows, cols)).float()
    
    # Iterate over each pixel. Images are generated one pixel
    # at a time
    for row in range(rows):
        for col in range(cols):
            for channel in range(channels):
                # Feed the current outputs through the model
                # Output: (N, rows, cols, channels)
                probs = model(out_img).permute(0, 2, 3, 1)
                
                # We only want the probabilities for the current pixel
                # Output: (N)
                probs = probs[:, row, col].squeeze()
                probs = torch.argmax(probs, dim=-1)
                
                # Add some noise in the model so that it can
                # actually generate an image
                # probs = torch.ceil(
                #     probs - torch.rand(1).cuda()
                # )
                
                # Get the discrete pixel value
                # new_vals = torch.round(probs)
                # new_vals = torch.clamp(probs, 0, 255)
                
                # Set the new pixel value
                out_img[:, channel, row, col] = probs
    
    # Show the image
    out_img = out_img.permute(0, 2, 3, 1)
    plt.imshow(out_img.detach().cpu().squeeze(0))
    plt.show()
    
    
    
if __name__ == '__main__':
    PixelCNN_test()