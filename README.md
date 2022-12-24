# Diffusion_models_from_scratch
Creating a diffusion model from scratch in PyTorch to learn exactly how they work.

# Current Additions
This section will go over the current additions to the model. It starts with the basics of a diffusion model and goes into what I've added to improve it. Most of it will be according to [Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/abs/2105.05233).

## Basic Diffusion Model

A diffusion model is defined by a markov chain noising and unnoising process in which a noise scheduler, q, noises data from time t-1 to time t (forward process) and a learned function, p, unoises the data from time t to time t-1 (reverse process). The image below shows this process.

<img width="563" alt="image" src="https://user-images.githubusercontent.com/43501738/209416054-b2c11fc9-0887-4337-a314-f064bc9047d7.png">

## Scheduling the Forward Process

To noise the image from time t-1 to time t, a linear or cosine scheduler (proposed by the improved DDPM paper) can be used. A linear scheduler increases the varaince of noise added to the image constantly over timesteps while a cosine scheduler follows a more exponential path.

![image](https://user-images.githubusercontent.com/43501738/209417038-0ab559d8-61b6-43c5-aa65-d2f3c043839f.png)

The formula for the linear scheduler is defined as a linear set of values between two values. In the case of the original DDPM paper, they set the range to be from 1e-4 and 0. The scheduler defines the variance (beta_t) values, but using the beta_t variances, the authors control how the image is noised using the following equations:

<img width="518" alt="image" src="https://user-images.githubusercontent.com/43501738/209417185-cc9ca943-9ab2-4428-a87a-a94a66fe4039.png">

The formula for the cosine scheduler is defined by cumulative products of 1-beta values as a_bar_t:

<img width="440" alt="image" src="https://user-images.githubusercontent.com/43501738/209417238-dfbdda12-c531-4363-8224-58b68c654de7.png">
<img width="299" alt="image" src="https://user-images.githubusercontent.com/43501738/209417247-1bbe4332-5c72-4e54-bfbc-8729a06bdfc1.png">


## Modeling the Reverse Process

The model being used is a probabilistic U-net model in which the current noised image (of shape CxLxW) at time t-1 and the current timestep (t) is fed into the model. The output of the model is the same shape, but with double the number of channels (of shape 2CxLxW). The first half of the channels correspond to the noise predictions for each pixel: ε. That is, what is the noise that was added to the image at that timestep? The second half of the channels corespond to the predicted variances for each pixel: Σ. This value will be used when adding more noise to the image in the generative reverse process.

![image](https://user-images.githubusercontent.com/43501738/209416677-47dd1cd5-80d6-4191-92e7-ab3c127b6d2b.png)



## Faster inference With DDIM

The main problem with DDPMs is that they are very slow to generate as DDPMs take T steps to generate images where T is in the thousands. The DDIM paper attempts to fix this issue by essentially skipping steps from the diffusion process when generating images. To do this, DDIMs use the exact same training process as DDPMs, but only change the inference procedure to use a non-Markovian process instead of the default Markovian one. The Markovian process is inhenintly defined as a chain of sequential operations, but the non-Markovian process defined for DDIMs only use the original gaussian noise, allowing it to take large steps as they are not needed to generate the image, only to (usually) fine-tune it.

<img width="626" alt="image" src="https://user-images.githubusercontent.com/43501738/209421133-b75eaf0a-bf69-4912-a9c4-e314ef7b08eb.png">

To implement this non-Markovian definition of the generative process, the authors propose a change in the generation procedure:

<img width="638" alt="image" src="https://user-images.githubusercontent.com/43501738/209421172-4a514636-2b71-494f-acdb-de6c387f96e7.png">

With this change generation procedure, the model becomes both Markovian or non-Markovian by changing the value of σ. To easily change the value of σ, the authors propose the use of the following formulas:

<img width="697" alt="image" src="https://user-images.githubusercontent.com/43501738/209422771-dfe2c91c-fb3b-483d-86bb-35a14cf19eff.png">

Note that the proposed formulation of σ is the same as the hyperparameter η multiplied by the square root of the beta tilde value defined in the DDPM and improved DDPM paper. Using this formula, the generative process becomes a default Markovian DDPM when η is 1 and a deterministic DDIM when η is 0. τ is another hyperparameter which is the chain of t values the model will use to generate an image where τ is a subset of the set of all t values: [1, 2, ..., T-1, T]. This way, we can take some sort of subset of t values and generate an image in much fewer steps than usual.

Since the standard deviation is just the same as the static statndard deviation used by the DDPM authors and our model predicts Σ which ranges between beta and beta tilde, we can replace it with the sqare root of the predicted variance, Σ, of our model. This way the model still has control over the variance. In fact, this is similar to what the improved DDPM authors do. Instead, in this repo the model variance still controls the variance of the noise, but not the variance of the prediction. The variance of the x_t direction prediction is kept as the static σ value above. This way, the model can be changed from a DDPM to a DDIM using σ, but still has control over how the noise is added in the generation process with Σ. The formulation of the process is as follows:

![image](https://user-images.githubusercontent.com/43501738/209422729-ccfa8481-70ea-4cad-8a6a-c079e7dd1c3a.png)


## Classifier Free Guidance

[I will also attempt to implement this soon]
https://arxiv.org/abs/2207.12598




# Data
## ImageNet
https://www.kaggle.com/c/imagenet-object-localization-challenge/overview/description

## Mini-ImageNet
Used for creating the model on my laptop

https://www.kaggle.com/datasets/whitemoon/miniimagenet?select=mini-imagenet-cache-train.pkl

## Cifar-10 and Cifar-100
https://www.cs.toronto.edu/~kriz/cifar.html




# References

1. Main Paper Reference (diffusion models beat gans on image synthesis): https://arxiv.org/abs/2105.05233

2. Denoising Diffusion Probabilities Models (DDPMs): https://arxiv.org/abs/2006.11239

3. Improved DDPMs (Improved Denoising Diffusion Probabilistic Models): https://arxiv.org/abs/2102.09672

4. Denoising Diffusion Implicit Models (DDIM): https://arxiv.org/abs/2010.02502

5. U-net (Convolutional Networks for Biomedical Image Segmentation): https://arxiv.org/abs/1505.04597

6. ConvNext (A ConvNet for the 2020s): https://arxiv.org/abs/2201.03545

7. Where the BigGAN residual blocks comes from (Large Scale GAN Training for High Fidelity Natural Image Synthesis): https://arxiv.org/abs/1809.11096

8. What are non-local blocks? (Non-local Neural Networks): https://arxiv.org/pdf/1711.07971.pdf
