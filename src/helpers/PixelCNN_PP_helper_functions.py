import torch
import numpy as np





# Compute the CDF of the logistic distribution
# Input:
#   x - A pixel value to feed through the distribution
#   mu - Mu parameter of the logistic CDF function
#   scale - Scale factor for the logistic function
#   s_inv - The inverse of the s parameter of the logistic CDF function
def logistic_CDF(x, mu, scale, s_inv):
    return torch.sigmoid((x-mu+scale)*s_inv)




# Calculate the predicted values from the model given
# the model output
# Inputs:
#   mu - Mu parameter for the logistic distribution
#   s_inv - Inverse of the s parameter for the logistic distribution
#   pi - Weights for each of the K logistic distributions
#   scaled - True if the data is scaled between -1 and 1, False otherwise
# Output:
#   preds - Pixel value predictions of shape (N)
def get_preds(mu, s_inv, pi, scaled):
    
    # The weights are log probabilities, scale them
    # so that they are normal probabilities
    pi = torch.exp(pi)
    
    ### Calculate the probabilities for each pixel values
    # Scale factor for the logistic function
    scale = 1/(2*127.5) if scaled else 0.5
    
    # All pixel values to get the probability for
    # Shape is (N, 256, 5)
    pix_vals = torch.arange(0, 256).unsqueeze(0).unsqueeze(-1).repeat(mu.shape[0], 1, 5)
    
    # Reformat the value between -1 and 1 if needed
    if scaled:
        pix_vals = (pix_vals - 127.5)/127.5
        
        
        
    # Logistic value calculations
    # Cases: if the pixel value is -1 (0), 1 (255), or other
    pix_vals[:, :1] = logistic_CDF(pix_vals[:, :1], mu, scale, s_inv)
    pix_vals[:, -1:] = 1 - logistic_CDF(pix_vals[:, -1:], mu, -scale, s_inv)
    pix_vals[:, 1:-1] = logistic_CDF(pix_vals[:, 1:-1], mu, scale, s_inv) - \
        logistic_CDF(pix_vals[:, 1:-1], mu, -scale, s_inv)
        
    # Weigh the probabilities and sum them up
    # Shapes: (Nx256xK) -> (Nx256)
    pix_vals = pi.unsqueeze(1)*pix_vals
    pix_vals = pix_vals.sum(-1)
    
    # Make the probabilities between 0 and 1
    pix_vals /= torch.max(pix_vals.sum(-1), torch.tensor(1e-6))
    
    
    
    
    ### Now that the probabilities are calculated, create a
    ### multinomial distribution over all 256 pixel values
    ### and sample the distribution
    preds = torch.multinomial(pix_vals, 1)
    
    # If the values are scaled, scale the prediction
    if scaled:
        preds = ((preds - 127.5)/127.5)
    
    # Return the predictions
    return preds
