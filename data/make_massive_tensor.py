import torch
import os
import pickle







@torch.no_grad()
def main():
    # Directory name should be "data/Imagenet64" relative to root
    data_dir = "data" + os.sep + "Imagenet64"

    # Limit on the number of data to load (-1 for all)
    limit = 100

    # Massive tensor is pre-initialized to 64x64 images
    # of type uint8 and labels of type int 32
    imgs = torch.zeros(limit, 3, 64, 64, dtype=torch.uint8)
    labels = torch.zeros(limit, dtype=torch.int32)
    cur_iter = 0

    # Iterate over all files in the data directory
    for file in os.listdir(data_dir):
        # Get the full filename
        filename = data_dir + os.sep + file

        # Read in the file
        try:
            tens = pickle.load(open(filename, "rb"))
        except pickle.UnpicklingError:
            print(f"Error with file {filename}")
            continue

        # Get the image and label data
        img = tens["img"]
        label = tens["label"]

        # Convert the imgage and label to the corrent form
        img = torch.tensor(img, dtype=torch.uint8).reshape(3, 64, 64)
        label = torch.tensor(label, dtype=torch.int32)

        # Save the data to the massive tensors
        imgs[cur_iter] = img
        labels[cur_iter] = label

        # Increase the iteration count
        cur_iter += 1

        if cur_iter == limit:
            break
    
    # Save the massive tensors
    torch.save(imgs, "data/Imagenet64_imgs.pt")
    torch.save(labels, "data/Imagenet64_labels.pt")




if __name__ == "__main__":
    main()