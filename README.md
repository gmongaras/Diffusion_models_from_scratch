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

## Training

To run the training script make sure to ...

...

...

...

...

After the above is complete, you can run the training script as follows from the root directory of this repo:

`torchrun --nproc_per_node=[num_gpus] src/train.py --[params]`
- [num_gpus] is replaced by the number of desired GPUs to parallelize training on
- [params] is replaced with any of the parameters listed below

For example:

`torchrun --nproc_per_node=8 src/train.py --blk_types res,res,clsAtn,chnAtn --batchSize 32`

The above example runs the code with the following parameters:
- Run on 8 parallel GPUs.
- Each u-net block is composed of res->res->clsAtn->chnAtn sequential blocks.
- batchSize is 32 on each GPU, so a total batch size of 8*32 = 256.

`torchrun --nproc_per_node=1 src/train.py --loadModel True --loadDir models/models_res --loadFile model_479e_600000s.pkl --optimFile optim_479e_600000s.pkl --loadDefFile model_params_479e_600000s.json --gradAccSteps 2`

The above example loads in a pre-trained model for checkpoint:
- Use 1 GPU
- Load the model file `model_479e_600000s.pkl`
- Load the optimizer file `optim_479e_600000s.pkl`
- Load the model metadata file `model_params_479e_600000s.json`
- Use 2 gradient accumulation steps



The parameters of the script are as follows:

<b>Data Parameters</b>
- inCh [3] - Number of input channels for the input data.
- data_path [data/Imagenet64] - Path to the ImageNet 64x64 dataset.
- load_into_mem [True] - True to load all ImageNet data into memory, False to load the data from disk as needed.

<b>Model Parameters</b>
- embCh [128] - Number of channels in the top layer of the U-net. Note, this is scaled by 2^(chMult*layer) each U-net layer
- chMult [1] - At each U-net layer, at what scale should the channels be multiplied by? Each layer has embCh*2^(chMult*layer) channels
- num_layers [3] - Number of U-net layers. A value of 3 has a depth of 3, meaning there are 3 down layers and 3 up layers in the U-net
- blk_types [res,clsAtn,chnAtn] - How should the residual block be structured? (list of `res`, `conv`, `clsAtn`, `atn`, and/or `chnAtn`. 
  - res: Normal residual block. 
  - conv: ConvNext block.
  - clsAtn: Attention block to include extra class information (Q, K are cls features). 
  - atn: Attention ViT block for the hidden features.
  - clsAtn: Efficient-lightweight attention block over the feature channels. 
  - EX: `res,res,conv,clsAtn,chnAtn`
- T [1000] - Number of timesteps in the diffusion process
- beta_sched [cosine] - Noise scheduler to use in the diffusion process. Can be either `linear` or `cosine`
- t_dim [512] - Dimension of the vector encoding for the time information.
- c_dim [512] - Dimension of the vector encoding for the class information. NOTE: Use -1 for no class information
- atn_resolution [16] - Resolution of the attention block (atn). The resolution splits the image into patches of that resolution to act as vectors. Ex: a resolution of 16 creates 16x16 patches and flattens them as feature vectors.

<b>Training Parameters</b>
- Lambda [0.001] - Weighting term between the variance and mean loss in the model.
- batchSize [128] - Batch size on a single GPU. If using multiple GPUs, this batch size will be multiplied by the GPU count.
- gradAccSteps [1] - Number of steps to breakup the batchSize into. Instead of taking 1 massive step where the whole batch is loaded into memory, the batchSize is broken up into sizes of batchSize//numSteps so that it can fit into memory. Mathematically, the update will be the same, as a single batch update, but the update is distributed across smaller updates to fit into memory. A higher value takes more time, but uses less memory.
- device [gpu] - Device to put the model on. Use \"gpu\" to put it on one or multiple Cuda devices or \"cpu\" to put it on the CPU. CPU should only be used for testing.
- epochs [1000000] - Number of epochs to train for.
- lr [0.0003] - Model learning rate.
- p_uncond [0.2] - Probability of training on a null class for classifier-free guidance. Note that good values are 0.1 or 0.2. (only used if c_dim is not None)
- use_importance [False] - True to use importance sampling for values of t, False to use uniform sampling.

<b>Saving Parameters</b>
- saveDir [models/] - Directory to save models checkpoints to. NOTE that three files will be saved: the model .pkl file, the model metadata .json file, and the optimizer .pkl file for training reloading
- numSaveSteps [10000] -"Number of steps until a new model checkpoint is saved. This is not the number of epochs, rather it's the number of time the model has updates. NOTE that three files will be saved: the model .pkl file, the model metadata .json file, and the optimizer .pkl file for training reloading.

<b>Model loading Parameters</b>
- loadModel [False] - True to load a pretrained model from a checkpoint. False to use a randomly initialized model. Note that all three model files are needed for a successfull restart: the model .pkl file, the model metadata .json file, and the optimizer .pkl file.
- loadDir [models/] - Directory of the model files to load in.
- loadFile [""] - Model .pkl filename to load in. Will looks something like: model_10e_100s.pkl
- optimFile [""] - Optimizer .pkl filename to load in. Will looks something like: optim_10e_100s.pkl
- loadDefFile [""] - Model metadata .json filename to load in. Will looks something like: model_params_10e_100s.json

<b>Data loading parameters</b>
- reshapeType [""] - If the data is unequal in size, use this to reshape images up by a power of 2, down a power of 2, or not at all ("up", "down", "")



## Inference

To run inference be sure too...

...

...

...

...

...

After the above is done, you can run the script as follows from the root directory of this repo:

`python -m src.infer --loadDir [Directory location of models] --loadFile [Filename of the .pkl model file] --loadDefFile [Filename of the .json model parameter file] --[other params]`

For example, if I downloaded the model_358e_450000s file for the models_res_res_atn model and I want to use my CPU with a step size of 20, I would use the following on the command line:

`python -m src.infer --loadDir models/models_res_res_atn --loadFile model_358e_450000s.pkl --loadDefFile model_params_358e_450000s.json --device cpu --step_size 20`

The parameters of the inference scripts are as follows:

<b>Required</b>:
- loadDir - Location of the models to load in.
- loadFile - Name of the .pkl model file to load in. Ex: model_358e_450000s.pkl
- loadDefFile - Name of the .json model file to load in. Ex: model_params_358e_450000s.pkl

<b>Generation parameters</b>
- step_size [10] - Step size when generating. A step size of 10 with a model trained on 1000 steps takes 100 steps to generate. Lower is faster, but produces lower quality images.
- DDIM_scale [0] - Must be >= 0. When this value is 0, DDIM is used. When this value is 1, DDPM is used. A low scalar performs better with a high step size and a high scalar performs better with a low step size.
- device ["gpu"] - Device to put the model on. use "gpu" or "cpu".
- guidance [4] - Classifier guidance scale which must be >= 0. The higher the value, the better the image quality, but the lower the image diversity.
- class_label [0] - 0-indexed class value. Use -1 for a random class and any other class value >= 0 for the other classes. FOr imagenet, the class value range from 0 to 999 and can be found in data/class_information.txt
- corrected [False] - True to put a limit on generation, False to not put a litmit on generation. If the model is generating images of a single color, then you may need to set this flag to True. Note: This restriction is usually needed when generating long sequences (low step size) Note: With a higher guidance w, the correction usually messes up generation.

<b>Output parameters</b>
- out_imgname ["fig.png"] - Name of the file to save the output image to.
- out_gifname ["diffusion.gif"] - Name of the file to save the output image to.
- gif_fps [10] - FPS for the output gif.




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

8. Attention block (Attention Is All You Need): https://arxiv.org/abs/1706.03762

9. Attention/Vit block (An Image is Worth 16x16 Words): https://arxiv.org/abs/2010.11929

10. Channel Attention block (ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks): https://arxiv.org/abs/1910.03151

Thanks to the following link for helping me multi-gpu the project!
https://theaisummer.com/distributed-training-pytorch/

Thanks to Huggingface for the Residual Blocks!
https://huggingface.co/blog/annotated-diffusion#resnet-block
