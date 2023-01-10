from torch.utils.data import Dataset
import torch
from helpers.image_rescale import reduce_image




class CustomDataset(Dataset):
    """Generative Dataset."""

    def __init__(self, data, classes=None, transform=True):
        """
        Args:
            image_data (tensor of tensors): Loaded dataset as a tensor
            classes (tensor of tensors): Data labels as a tensor
            transform (boolean): Transform data between -1 and 1
        """
        self.data = data
        self.classes = None if classes is None else classes.to(torch.int)

        # If the data is not in the range -1 to 1, change it
        # to this range
        if transform == True and self.data.max() > torch.tensor([1]):
            self.data = reduce_image(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], None if self.classes is None else self.classes[idx]