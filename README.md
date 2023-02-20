# Contents
- Current Additions
  - Basic Diffusion Model (DDPM)
  - Scheduling the Forward Process
  - Modeling the Reverse Process
  - Faster Inference with DDIMs
  - Classifier Free Guidance
- Environment Setup
- Train A Model
- Generate Images With Pretrained Models
- Calculating FID for a pretrained model
- My Results
- Data
- References


# Current Additions
This section will go over the current additions to the model. It starts with the basics of a diffusion model and goes into what I've added to improve it. Most of it will be according to [Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/abs/2105.05233).

## Basic Diffusion Model (DDPM)

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



## Faster inference With DDIMs

The main problem with DDPMs is that they are very slow to generate as DDPMs take T steps to generate images where T is in the thousands. The DDIM paper attempts to fix this issue by essentially skipping steps from the diffusion process when generating images. To do this, DDIMs use the exact same training process as DDPMs, but only change the inference procedure to use a non-Markovian process instead of the default Markovian one. The Markovian process is inhenintly defined as a chain of sequential operations, but the non-Markovian process defined for DDIMs only use the original gaussian noise, allowing it to take large steps as they are not needed to generate the image, only to (usually) fine-tune it.

<img width="626" alt="image" src="https://user-images.githubusercontent.com/43501738/209421133-b75eaf0a-bf69-4912-a9c4-e314ef7b08eb.png">

To implement this non-Markovian definition of the generative process, the authors propose a change in the generation procedure:

<img width="638" alt="image" src="https://user-images.githubusercontent.com/43501738/209421172-4a514636-2b71-494f-acdb-de6c387f96e7.png">

With this change generation procedure, the model becomes both Markovian or non-Markovian by changing the value of σ. To easily change the value of σ, the authors propose the use of the following formulas:

<img width="697" alt="image" src="https://user-images.githubusercontent.com/43501738/209422771-dfe2c91c-fb3b-483d-86bb-35a14cf19eff.png">

Note that the proposed formulation of σ is the same as the hyperparameter η multiplied by the square root of the beta tilde value defined in the DDPM and improved DDPM paper. Using this formula, the generative process becomes a default Markovian DDPM when η is 1 and a deterministic DDIM when η is 0. τ is another hyperparameter which is the chain of t values the model will use to generate an image where τ is a subset of the set of all t values: [1, 2, ..., T-1, T]. This way, we can take some sort of subset of t values and generate an image in much fewer steps than usual.

Since the standard deviation is just the same as the static standard deviation used by the DDPM authors and our model predicts Σ which ranges between beta and beta tilde, we can replace it with the sqare root of the predicted variance, Σ, of our model. This way the model still has control over the variance. In fact, this is similar to what the improved DDPM authors do. Instead, in this repo the model variance still controls the variance of the noise, but not the variance of the prediction. The variance of the x_t direction prediction is kept as the static σ value above. This way, the model can be changed from a DDPM to a DDIM using σ, but still has control over how the noise is added in the generation process with Σ. The formulation of the process is as follows:

![image](https://user-images.githubusercontent.com/43501738/209422729-ccfa8481-70ea-4cad-8a6a-c079e7dd1c3a.png)


## Classifier Free Guidance

Before classifier-free guidance, classifier-guidance was proposed to 

[I will also attempt to implement this soon]
https://arxiv.org/abs/2207.12598






# Environment Setup

First, download the data from this repo using the following on the command line

```
git clone https://github.com/gmongaras/Diffusion_models_from_scratch.git
cd Diffusion_models_from_scratch/
```

(Optional) If you don't want to change your environment, you can first create a virtual environment:
```
pip install virtualenv
python -m venv MyEnv/
```
Activate the virtual environment: [https://docs.python.org/3/library/venv.html#how-venvs-work](https://docs.python.org/3/library/venv.html#how-venvs-work)

Windows: `MyEnv\Scripts\activate.bat`

Linux: `source MyEnv/bin/activate`



Before running any scripts, make sure to download the correct packages and package versions. You can do so by running the following commands to upgrade pip and install the necessary package versions:
```
pip install pip -U
pip install -U requirements.txt
```

Note: PyTorch should be installed with cuda enabled if training and probably should have cuda if generating images, but is not required. The cuda version downloaded may be different from the one needed. The cuda versions and how to download them can be found below:

[https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

Now the enviroment should be setup properly.




# Downloading Data

Imagenet data can be downloaded from the following link:
https://image-net.org/download-images.php

To get the data, you must first request access and be accepted to download the Imagenet data. I trained my models on Imagenet 64x64

<img width="326" alt="image" src="https://user-images.githubusercontent.com/43501738/219979158-829771f5-6357-4e21-acc5-b0c903ac1557.png">

Once downloaded, you should pur both the `Imagenet64_train_part1.zip` and `Imagenet64_train_part2.zip` in the data/ directory.

The zip files are in the correct directory, run the following script to load the data into the necessary format:

`python data/loadImagenet64.py`

If you wish to load the data into memory before training, run the script below. Otherwise, the data will be extracted from disk as needed.

`python data/make_massive_tensor.py`


The directory should look as follows when all data is downloaded: [Directory Structure](#directory-structure)





# Directory Structure

```
.
├── data
│   ├── Imagenet64
|   |   ├── 0.pkl
|   |   ├── ...
|   |   ├── metadata.pkl
│   ├── Imagenet64_train_part1.zip
│   ├── Imagenet64_train_part1.zip
│   ├── README.md
│   ├── archive.zip
│   ├── loadImagenet64.py
│   ├── make_massive_tensor.py
├── eval
|   ├── __init__.py
|   ├── compute_FID.py
|   ├── compute_imagenet_stats.py
|   ├── compute_model_stats.py
|   ├── compute_model_stats_multiple.py
├── models
|   ├── README.md
|   ├── [model_param_name].json
|   ├── [model_name].pkl
├── src
|   ├── blocks
|   |   ├── BigGAN_Res.py
|   |   ├── BigGAN_ResDown.py
|   |   ├── BigGAN_ResUp.py
|   |   ├── ConditionalBatchNorm2D.py
|   |   ├── Efficient_Channel_Attention.py
|   |   ├── Multihead_Attn.py
|   |   ├── Non_local.py
|   |   ├── Non_local_MH.py
|   |   ├── PositionalEncoding.py
|   |   ├── Spatial_Channel_Attention.py
|   |   ├── __init__.py
|   |   ├── clsAttn.py
|   |   ├── convNext.py
|   |   ├── resBlock.py
|   |   ├── wideResNet.py
|   ├── helpers
|   |   ├── PixelCNN_PP_helper_functions.py
|   |   ├── PixelCNN_PP_loss.py
|   |   ├── image_rescale.py
|   |   ├── multi_gpu_helpers.py
|   ├── models
|   |   ├── PixelCNN.py
|   |   ├── PixelCNN_PP.py
|   |   ├── U_Net.py
|   |   ├── Variance_Scheduler.py
|   |   ├── diff_model.py
|   ├── CustomDataset.py
|   ├── __init__.py
|   ├── infer.py
|   ├── model_trainer.py
|   ├── train.py
├── tests
|   ├── BigGAN_Res_test.py
|   ├── U_Net_test.py
|   ├── __init__.py
|   ├── diff_model_noise_test.py
├── .gitattributes
├── .gitignore
├── README.md
```







# Train A Model

Before training a model, make sure you [setup the environment] and [downloaded the data]

To train a model, run the following command from the root directory.
`torchrun --nproc_per_node={n_gpus} src/train.py`

Replace `n_gpus` with the number of desired GPUs to use.


## Training Parameters

params



# Generate Images With Pretrained Models





# Calculating FID for a pretrained model





# My Results








# Usage

To run the train script, use the following command from the root directory:
`torchrun --standalone --nnodes=1 --nproc_per_node=[num_gpus] src/train.py`
- num_gpus - The number of gpus to split he model onto

To run the infer script, use the following command from the root directory:
`python -m src.infer`




# Data
## ImageNet (64x64)
https://image-net.org/download-images.php




# References

1. Diffusion Models Beat GANs on Image Synthesis (with classifier guidance): https://arxiv.org/abs/2105.05233

2. Denoising Diffusion Probabilities Models (DDPMs): https://arxiv.org/abs/2006.11239

3. Improved DDPMs (Improved Denoising Diffusion Probabilistic Models): https://arxiv.org/abs/2102.09672

4. Denoising Diffusion Implicit Models (DDIM): https://arxiv.org/abs/2010.02502

5. Classifier-Free Guidance: https://arxiv.org/abs/2207.12598

6. U-net (Convolutional Networks for Biomedical Image Segmentation): https://arxiv.org/abs/1505.04597

7. ConvNext (A ConvNet for the 2020s): https://arxiv.org/abs/2201.03545

Thanks to the following link for helping me multi-gpu the project!
https://theaisummer.com/distributed-training-pytorch/

Thanks to Huggingface for the Residual Blocks!
https://huggingface.co/blog/annotated-diffusion#resnet-block
