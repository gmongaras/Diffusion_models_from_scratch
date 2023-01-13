import torch
from torch import nn
from torchvision import transforms
import numpy as np
from scipy.linalg import sqrtm



def compute_FID():
    # Given a two files of means and two files of
    # variances, compute the FID of these two distributions


    # Mean and variance files
    mean_file1 = "eval/saved_stats/real_mean.npy"
    mean_file2 = "eval/saved_stats/fake_mean_10K.npy"
    var_file1 = "eval/saved_stats/real_var.npy"
    var_file2 = "eval/saved_stats/fake_var_10K.npy"


    # Load in the mean and variances
    mean1 = np.load(mean_file1)
    mean2 = np.load(mean_file2)
    var1 = np.load(var_file1)
    var2 = np.load(var_file2)




    # calculate sum squared difference between means
    ssdiff = np.sum((mean1 - mean2)**2.0)

	# calculate sqrt of product between cov
    covmean = sqrtm(var1.dot(var2))

	# check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real

	# calculate score
    fid = ssdiff + np.trace(var1 + var2 - 2.0 * covmean)
    return fid









if __name__ == "__main__":
    print(compute_FID())