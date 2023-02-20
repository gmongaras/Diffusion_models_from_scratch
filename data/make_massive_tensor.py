import torch
import os
import pickle







@torch.no_grad()
def main():
    # Directory name should be "data/Imagenet64" relative to root
    data_dir = "data" + os.sep + "Imagenet64"

    # Limit on the number of data to load (-1 for all)
    limit = 100

    



    # If the limit is -1, we want to see how many files there
    # are and assume that is the limit
    if limit == -1:
        limit = len(os.listdir(data_dir))

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
            print(f"Pickle error with file {filename}")
            continue

        # Get the image and label data
        try:
            img = tens["img"]
            label = tens["label"]
        except KeyError:
            print(f"Key error with file {filename}")
            continue

        # Convert the imgage and label to the corrent form
        try:
            img = torch.tensor(img, dtype=torch.uint8).reshape(3, 64, 64)
            label = torch.tensor(label, dtype=torch.int32)
        except RuntimeError:
            print(f"Shape error with file {filename}")
            continue

        # Save the data to the massive tensors
        imgs[cur_iter] = img
        labels[cur_iter] = label

        # Increase the iteration count
        cur_iter += 1

        if cur_iter == limit:
            break
    
    # If the limit is -1, there may have been some empty
    # files, so we want to slice by the cur_iter value
    # which represents the actual number of images loaded in
    imgs = imgs[:cur_iter]
    labels = labels[:cur_iter]
    
    # Save the massive tensors
    torch.save(imgs, "data/Imagenet64_imgs.pt")
    torch.save(labels, "data/Imagenet64_labels.pt")




if __name__ == "__main__":
    main()