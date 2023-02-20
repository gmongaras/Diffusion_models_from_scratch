from torch.utils.data import Dataset
import torch
from helpers.image_rescale import reduce_image
import pickle
import os
import numpy as np
import math




class CustomDataset(Dataset):
    """Generative Dataset."""

    def __init__(self, data_path, num_data, cls_min, transform=True, shuffle=True, scale=None, loadMem=False):
        """
        Args:
            data_path (str): Path to the data to load in
            num_data (int): Total number of data points to train the model on
            cls_min (int): The min class value in the data
            transform (boolean): Transform data between -1 and 1
            shuffle (boolean): True to shuffle the data upon entering. False otherwise
            scale (str or NoneType): Scale data "up" or "down" to the nearest power of 2
                                     or keep the data the same shape with None
            loadMem (boolean): True to load in all data to memory, False to keep it on disk
        """

        # Save the data information
        self.data_path = data_path
        self.num_data = num_data
        self.transform = transform
        self.scale = scale
        self.loadMem = loadMem

        # The min class value represents the value that needs to be
        # subtracted from the class value so the min value will be 0
        self.cls_scale = cls_min

        # Load in all the data onto the disk if specified
        if self.loadMem:
            # Load in the massive data tensors
            self.data_mat = torch.load("data/Imagenet64_imgs.pt")
            self.label_mat = torch.load("data/Imagenet64_labels.pt")

            # Get the number of data loaded in
            self.num_data = self.data_mat.shape[0]

            # Make sure the labels and data have the same shapes
            assert self.data_mat.shape[0] == self.label_mat.shape[0]

            print(f"{self.num_data} data loaded in")

        
        # Create a list of indices which can be used to
        # essentially shuffle the data
        self.data_idxs = np.arange(0, self.num_data)
        if shuffle:
            np.random.shuffle(self.data_idxs)


        
    def __len__(self):
        return self.num_data

    def __getitem__(self, idx):

        # Convert the given index to the shuffled index
        data_idx = self.data_idxs[idx]

        # If the files were pre-loaded into memory,
        # just grab them from meory
        if self.loadMem == True:
            image = self.data_mat[data_idx].clone()
            label = self.label_mat[data_idx].clone()

            # Subtract the min class value so the min label is 0
            label -= self.cls_scale
        
        # If the files are not preloaded, then
        # get them from disk individually
        else:
            # Open the data file and load it in
            data = pickle.load(open(f"{self.data_path}{os.sep}{data_idx}.pkl", "rb"))

            # Get the image and class label from the data
            image = data["img"]
            label = data["label"]

            # Subtract the min class value so the min label is 0
            label -= self.cls_scale

            # Convert the data to a tensor
            image = torch.tensor(image, dtype=torch.float32, device=torch.device("cpu"))
            image = image.reshape(3, 64, 64)
            label = torch.tensor(label, dtype=torch.int)

        # Reshape the image to the nearest power of 2
        if self.scale is not None:
            if self.scale == "down":
                next_power_of_2 = 2**math.floor(math.log2(image.shape[-1]))
            elif self.scale == "up":
                next_power_of_2 = 2**math.ceil(math.log2(image.shape[-1]))
            image = torch.nn.functional.interpolate(image, (next_power_of_2, next_power_of_2))

        # Transform the image between -1 and 1
        if self.transform:
            image = reduce_image(image)

        # Return the image and label
        return image,label