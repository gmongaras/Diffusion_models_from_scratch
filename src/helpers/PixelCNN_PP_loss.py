import torch





# Loss for the PixelCNN++ model
# Note: The loss calculation comes from 4 inputs and 1 output.
# The inputs are pi, mu, and s_inv for the distribution. The foruth
# input is x, the pixel value. The output is the probability that
# the model thinks the pixel, x (the input), should be in the
# current location (which should be 100%). The goal is to have the model
# maximize the probability of x for each pixel in the image.
# Input:
#   Y_hat - Predictions from the model of shape (N, C=9K, L, W)
#   Y - True labels for the model of shape (N, C=3, L, W)
def PixelCNN_PP_loss(Y_hat, Y):
    # Shape assertions
    assert Y_hat.shape[0] == Y.shape[0] and\
        Y_hat.shape[2:] == Y.shape[2:], "Shapes not aligned properly"
        
    # Permute the data to have channels as the last dimension
    Y_hat = Y_hat.permute(0, 2, 3, 1)
    Y = Y.permute(0, 2, 3, 1)
    
    # Reshape the predictions to distinguish the channels
    # (N, L, W, 9K) -> (N, L, W, 3, 3K)
    Y_hat = Y_hat.reshape(*Y_hat.shape[:-1], 3, Y_hat.shape[-1]//3)
    
    # Get mu, s_inv, and pi from the predictions
    # Each have shape (N, L, W, 3, K)
    mu_hat = Y_hat[:, :, :, :, ::3]
    s_inv_hat = Y_hat[:, :, :, :, 1::3]
    pi_hat = Y_hat[:, :, :, :, 2::3]
    
    
    
    # What scale is being used? -1 to 1 or 0 to 255?
    # True for -1 to 1, False for 0 to 255
    scaled = True if round(Y.max().item()) == 1.0 else False
    
    # Scale factor for the loss function
    scale = 1/(2*127.5) if scaled else 0.5
    
    
    #### 4 Cases:
    # 1. Black pixel (x <= 0) or (x < -0.999)
    # 2. White pixel (x >= 255) or (x > 0.999)
    # 3. Overflow condition (small difference leading to NaN due to log)
    # 4. Normal case
    
    ## Case 1: Black pixel
    
    # Where is this condition true?
    C1 = torch.where(Y < -0.999 if scaled else Y <= 0)
    
    # Get the X values and the predicted distribution
    # parameters from the data
    # (Parameters have shape (N, K))
    C1_mu = mu_hat[C1]
    C1_s_inv = s_inv_hat[C1]
    C1_pi = pi_hat[C1]
    C1_X = Y[C1].unsqueeze(-1)
    
    # Calculate the loss for these values
    C1_im = (C1_X - C1_mu + scale)*C1_s_inv
    # C1_loss = C1_im - torch.nn.functional.softplus(C1_im)
    C1_loss = torch.log(torch.sigmoid(C1_im))
    
    # Scale the loss for each distribution and get the final
    # loss value
    # (Loss had shape (N, K), now has shape (N))
    C1_loss = (C1_pi + C1_loss).sum(-1)
    
    
    ## Case 2: White pixel
    C2 = torch.where(Y > 0.999 if scaled else Y >= 255)
    
    # Get the X values and the predicted distribution
    # parameters from the data
    # (Parameters have shape (N, K))
    C2_mu = mu_hat[C2]
    C2_s_inv = s_inv_hat[C2]
    C2_pi = pi_hat[C2]
    C2_X = Y[C2].unsqueeze(-1)
    
    # Calculate the loss for these values
    C2_im = (C2_X - C2_mu - scale)*C2_s_inv
    # C2_loss = -torch.nn.functional.softplus(C2_im)
    C2_loss = torch.log(1-torch.sigmoid(C2_im))
    
    # Scale the loss for each distribution and get the final
    # loss value
    # (Loss had shape (N, K), now has shape (N))
    C2_loss = (C2_pi + C2_loss).sum(-1)
    
    
    ## Case 3: Overflow condition
    C3_t1 = torch.sigmoid((Y.unsqueeze(-1) - mu_hat + scale)*s_inv_hat)
    C3_t2 = torch.sigmoid((Y.unsqueeze(-1) - mu_hat - scale)*s_inv_hat)
    C3_c = (pi_hat*(C3_t1 - C3_t2)).sum(-1)
    C3 = torch.where(torch.logical_and(
        C3_c < 10e-5,
        ~torch.logical_or( # We don't want to redo C1 and C2 losses
            Y < -0.999 if scaled else Y <= 0,
            Y > 0.999 if scaled else Y >= 255
        )
    ))
    
    # Get the X values and the predicted distribution
    # parameters from the data
    # (Parameters have shape (N, K))
    C3_mu = mu_hat[C3]
    C3_s_inv = s_inv_hat[C3]
    C3_pi = pi_hat[C3]
    C3_X = Y[C3].unsqueeze(-1)
    
    # Calculate the loss for these values
    C3_im = (C3_X - C3_mu) * C3_s_inv
    C3_loss = -C3_im - torch.log(1/C3_s_inv) \
        - 2*torch.nn.functional.softplus(-C3_im) \
        - torch.log(torch.tensor(127.5) if scaled else torch.tensor(1.0))
    
    # Scale the loss for each distribution and get the final
    # loss value
    # (Loss had shape (N, K), now has shape (N))
    C3_loss = (C3_pi + C3_loss).sum(-1)
    
    
    
    ## Case 4: Normal case
    
    C4 = torch.where(torch.logical_and(
        torch.logical_and(~(Y < -0.999 if scaled else Y <= 0),
                          ~(Y > 0.999 if scaled else Y >= 255)
        ),
        ~(C3_c < 10e-5)
    ))
    
    # Get the X values and the predicted distribution
    # parameters from the data
    # (Parameters have shape (N, K))
    C4_mu = mu_hat[C4]
    C4_s_inv = s_inv_hat[C4]
    C4_pi = pi_hat[C4]
    C4_X = Y[C4].unsqueeze(-1)
    
    # Calculate the loss for these values
    C4_im_1 = torch.sigmoid((C4_X - C4_mu + scale)*C4_s_inv)
    C4_im_2 = torch.sigmoid((C4_X - C4_mu - scale)*C4_s_inv)
    C4_loss = torch.log(
        torch.max(
            C4_im_1 - C4_im_2,
            torch.tensor(6e-6)
        )
    )
    
    # Scale the loss for each distribution and get the final
    # loss value
    # (Loss had shape (N, K), now has shape (N))
    C4_loss = (C4_pi + C4_loss).sum(-1)
    
    
    
    # Get the final loss
    combined = torch.cat((C1_loss, C2_loss, C3_loss, C4_loss))
    return -combined.sum()