import numpy as np
import zipfile
import pickle
import torch
from torch import nn
from torchvision import transforms
import math


cpu = torch.device("cpu")








def compute_imagenet_stats():
    # Find the mean and variance of the imagenet dataset


    # Paths to the imagenet datasets
    archive_file1 = "data/Imagenet64_train_part1.zip"
    archive_file2 = "data/Imagenet64_train_part2.zip"

    batchSize = 300
    num_imgs = 100000
    device = torch.device("cuda")




    # Open the zip files
    archive1 = zipfile.ZipFile(archive_file1, 'r')
    archive2 = zipfile.ZipFile(archive_file2, 'r')
    
    # Read the pickle data
    data = []
    for filename in archive1.filelist:
        file = pickle.load(archive1.open(filename.filename, "r"))
        data.append(file["data"])
        del file
    for filename in archive2.filelist:
        file = pickle.load(archive2.open(filename.filename, "r"))
        data.append(file["data"])
        del file
    
    # Load the data
    archive1.close()
    archive2.close()
    img_data = np.concatenate((data), axis=0)
    del data

    # Shuffle the data
    np.random.shuffle(img_data)

    # Convert the data to a tensor
    img_data = torch.tensor(img_data, dtype=torch.uint8, device=torch.device("cpu"))
    img_data = img_data.reshape(img_data.shape[0], 3, 64, 64)

    # Get a subset of the data
    img_data = img_data[:num_imgs]

    # Used to transforms the images to the correct distirbution
    # as shown here: https://pytorch.org/hub/pytorch_vision_inception_v3/
    def normalize(imgs):
        # Convert image to 299x299
        imgs = transforms.Compose([transforms.Resize((299,299))])(imgs)

        # Standardize to [0, 1]
        imgs = imgs/255.0

        # Normalize by mean and std
        return transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])(imgs)








    # Load in the inception network
    inceptionV3 = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', weights="Inception_V3_Weights.DEFAULT")
    inceptionV3.eval()
    inceptionV3.to(device)

    # Remove the fully connected output layer
    inceptionV3.fc = nn.Identity()
    inceptionV3.aux_logits = False







    # Calculate the inception scores
    scores = None
    with torch.no_grad():
        for i in range(math.ceil(num_imgs/batchSize)):
            # Get the current batch size
            cur_batch_size = min(num_imgs, batchSize*(i+1))-batchSize*i

            # Get the current batch of images
            imgs = img_data[batchSize*i:cur_batch_size+(batchSize*i)]

            # Normalize the inputs
            imgs = normalize(imgs.to(torch.uint8))

            # Calculate the inception scores and store them
            if type(scores) == type(None):
                scores = inceptionV3(imgs.to(device)).to(cpu).numpy().astype(np.float16)
            else:
                scores = np.concatenate((scores, inceptionV3(imgs.to(device)).to(cpu).numpy().astype(np.float16)), axis=0)
    

    # Delete the model as its no longer needed
    del inceptionV3

    # Calculate the mean and covariance of the generated batch
    mean = np.mean(scores, axis=0)
    var = np.cov(scores, rowvar=False)


    # Save the mean and variance
    np.save("eval/saved_stats/real_mean.npy", mean)
    np.save("eval/saved_stats/real_var.npy", var)







if __name__ == "__main__":
    compute_imagenet_stats()