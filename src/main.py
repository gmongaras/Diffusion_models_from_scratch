import zipfile
import pickle
import torch






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
    from .blocks.PixelCNN_Residual import PixelCNN_Residual
    a = PixelCNN_Residual(3, "B")
    a(torch.rand(32, 3, 16, 16))
    
    
    
    
    
if __name__ == '__main__':
    main()