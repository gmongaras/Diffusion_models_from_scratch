# Summary
This repo is composed of DDPM, DDIM, and Classifier-Free guided models trained on ImageNet 64x64. More information can be found below.

To go along with this repo, I also [wrote an article](https://medium.com/@gmongaras/diffusion-models-ddpms-ddims-and-classifier-free-guidance-e07b297b2869) explaining the algorithms behind it.

# Contents
- [Current Additions](#current-additions)
- [Environment Setup](#environment-setup)
- [Downloading Pre-Trained Models](#downloading-pre-trained-models)
- [Downloading Training Data](#downloading-training-data)
- [Directory Structure](#directory-structure)
- [Train A Model](#train-a-model)
- [Generate Images With Pretrained Models](#generate-images-with-pretrained-models)
- [Calculating FID for a pretrained model](#calculating-fid-for-a-pretrained-model)
- [My Results](#my-results)
- [References](#references)


# Current Additions

This repo has the following Diffusion features:
- Normal DDPM
- Improved DDPM with cosine scheduler and variance prediction
- DDIM for faster inference
- Classifier-Free Guidance to improve image quality

Instead of going into each of the parts here, you can read an [article I wrote](https://medium.com/@gmongaras/diffusion-models-ddpms-ddims-and-classifier-free-guidance-e07b297b2869) which explains each part in detail.




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




# Downloading Pre-Trained Models

## Pre-Trained Model Notes
I have several pre-trained models available to download or varying model architecture types. There are 5 model types based on the u-net block construction.
- res ➜ conv ➜ clsAtn ➜ chnAtn (Res-Conv)
- res ➜ clsAtn ➜ chnAtn (Res)
- res ➜ res ➜ clsAtn ➜ chnAtn (Res-Res)
- res ➜ res ➜ clsAtn ➜ atn ➜ chnAtn (Res-Res-Atn)
- res ➜ clsAtn ➜ chnAtn with 192 channels (Res Large)

The above notation comes from the [Train A Model](#train-a-model) section under the blk_types parameter.


Each model was trained with the following parameters unless otherwise specified:
1. Image Resolution: 64x64
2. Initial embedding channel: 128
2. Channel multiplier — 1
3. Number of U-net blocks — 3
4. Timesteps — 1000
5. VLB weighting Lambda — 0.001
6. Beta Scheduler — Cosine
7. Batch Size — 128 (across 8 GPUs, so 1024)
8. Gradient Accumulation Steps — 1
9. Number of steps (Note: This is not epochs, a step is a single gradient update to the model)— 600,000
10. Learning Rate — 3*10^-4 = 0.0003
11. Time embedding dimension size— 512
12. Class embedding dimension size — 512
13. Probability of null class for classifier-free guidance — 0.2
14. Attention resolution — 16

Below are some training notes:
- I used 8 parallel GPUs each with a batch size of 128. SO, the batch size is 8*128 = 1024.
- I trained each model for a total of 600,000 steps. Note that this isn't number of epochs, but rather model updates. The models look to be able to be trained for longer since the FID values look the be decreasing even at 600,000 steps if you wish to continue training from a pre-trained checkpoint.
- Training the smaller models (res-conv, res, res-res) took 6-7 days to train and the larger models took about 8 days to train on 8 A100s.


## Picking a Model
To pick a model, I suggest looking at the [results](#my-results). The lower the FID score, the better better the outputs of the model are. The best models according to the results are:
- `res-res-atn`:
  - 358 epochs (358e)
  - 450000 steps (450000s)
  - The files for this model are:
    - model file: `model_358e_450000s.pkl`
    - model file: `model_params_358e_450000s.json`
    - optim file: `optim_358e_450000s.pkl`
- `res-res`:
  - 438 epochs (438e)
  - 550000 steps (550000s)
  - The files for this model are:
    - model file: `model_438e_550000s.pkl`
    - model metadata: `model_params_438e_550000s.json`
    - optimizer: `optim_438e_550000s.pkl`
  
  
## Downloading A Model
Once the model has been picked, you can download a model at the following link:

[Google Drive link](https://drive.google.com/drive/folders/1NueH5GaDYAI_8G9GJ19G06ijDp1scX0O)

<b>For training from a checkpoint</b> you need to download three files for a model:
- The model .pkl file (ex: model_438e_550000s.pkl)
- The model metadata .json file (ex: model_params_438e_550000s.json)
- The optimizer .pkl file (ex: optim_438e_550000s.pkl)

<b>For inference/generation</b> you only need to download two files for the model:
- The model .pkl file (ex: model_438e_550000s.pkl)
- The model metadata .json file (ex: model_params_438e_550000s.json)

Put these files in the `models/` directory to easily load them in when training/generating.




# Downloading Training Data

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

If you download both pretrained models and the training data, your directory should look like the following tree.

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

<b>Before training a model, make sure you [setup the environment](#environment-setup) and [downloaded the data](#downloading-data)</b>

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
- gradAccSteps [1] - Number of steps to breakup the batchSize into. Instead of taking 1 massive step where the whole batch is loaded into memory, the batchSize is broken up into sizes of batchSize//gradAccSteps so that it can fit into memory. Mathematically, the update will be the same, as a single batch update, but the update is distributed across smaller updates to fit into memory. A higher value takes more time, but uses less memory.
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






# Generate Images With Pretrained Models

<b>Before training a model, make sure you [setup the environment](#environment-setup) and [downloaded pre-trained models](#downloading-pre-trained-models)</b>

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

<b>Note</b>: The class values and labels are zero-indexed and can be found in [this document](https://github.com/gmongaras/Diffusion_models_from_scratch/blob/main/data/class_information.txt).





# Calculating FID for a pretrained model

Once you have trained your models, you can evaluate them here using these scripts.

Note: All scripts for the section are located in the `eval/` directory.

Calculating FID requires three steps:

<b>1: Compute statistics for the ImageNet Data</b>

For this step, run the `compute_imagenet_stats.py` to compute the FID for the ImageNet dataset.

`python -m eval.compute_imagenet_stats`

This script has the following parameters:
- archive_file1 - Path to the first ImageNet 64x64 zip file.
- archive_file2 - Path to the second ImageNet 64x64 zip file.
- batchSize - Batch size to parallelize the statistics generation.
- num_imgs - Number of images to sample from the dataset to compute statistics for.
- device - Device to compute the statistics on.



<b>2: Compute statistics for pretrained models</b>

This step has two alternatives. If you wish to generate FID for a single pre-trained, model use the `compute_model_stats.py` like so:

`python -m eval.compute_model_stats`

This script has the following paramters (which can be accessed by editting the file):
- model_dirname - Directory of the pre-trained model to compute stats for.
- model_filename - Filename of the pre-trained model.
- model_params_filename - Filename of the metadata of the pre-trained model.
- device - Device to run the model inference on
- gpu_num - GPU number to run model inference on (use 0 if only 1 GPU)
- num_fake_imgs - Number of images to generate before calculating stats of the model. Note that a value less than 10,000 is not recommended as the stats will not be accurate.
- batchSize - Size of a batch of images to generate at the same time. A higher value speeds up the process, but requires more GPU memory.
- step_size - Step size of the diffusion model (>= 1). This step size reduces the generation procedure by a factor of `step_size`. If the model requires 1000 steps to generate a single image, but has a step size of 4, then it will take 1000/4 = 250 steps to generate one image. Note that a higher step size means faster generation, but also lower quality images.
- DDIM_scale - Use 0 for a DDIM and 1 for a DDPM (>= 0). More information on this is located in the training section.
- corrected - True to put a limit on generation, False to keep the limit off generation. If the model is producing all black or white images, then this limit is probably needed. A low step size usually requires a limit.
- file_path - Path to where the statistics files should be saved to.
- mean_filename - Filename to save the mean statistic to.
- var_filename - FIlename to save the variance statistics to.


If you want to generate FID on multiple models and have access to multiple GPUs, you can parallelize the process. The `compute_model_stats_multiple.py` allows for this parallelization and can be run with the following command:

`python -m eval.compute_model_stats_multiple`

Note: The number of items in each of the lists should be at most equal to the number of GPUs you wish to use.

This script has the following parameters which can be changed inside the script file:
- dir_name - Directory to load all model files from.
- model_filenames - List of model filenames to calculate FID for.
- model_params_filenames - List of model metadata filenames.
- gpu_nums - GPU numbers to put each model on. The index of the GPU number corresponds to the index of the filename.
- step_size - Step size of the diffusion model (>= 1). This step size reduces the generation procedure by a factor of `step_size`. If the model requires 1000 steps to generate a single image, but has a step size of 4, then it will take 1000/4 = 250 steps to generate one image. Note that a higher step size means faster generation, but also lower quality images.
- DDIM_scale - Use 0 for a DDIM and 1 for a DDPM (>= 0). More information on this is located in the training section.
- corrected - True to put a limit on generation, False to keep the limit off generation. If the model is producing all black or white images, then this limit is probably needed. A low step size usually requires a limit.
- num_fake_imgs - Number of images to generate before calculating stats of the model. Note that a value less than 10,000 is not recommended as the stats will not be accurate.
- batchSize - Size of a batch of images to generate at the same time. A higher value speeds up the process, but requires more GPU memory.
- file_path - Directory to save all model statistics to.
- mean_filenames - Filenames to save the mean statistic for each model to.
- var_filenames - Filenames to save the variance statistic of each model to.

<b>Note</b>: Compared to the first step, this step is much more computationally heavy as it reqires the generation of images. Since it's a diffusion model, it has the downside of having to generate T (1000) images before a single image is even generated.


<b>3: Compute the FID between ImageNet and the model(s)</b>

Once you have generated both the FID and ImageNet statistics, you can compute the FID scores using the `compute_FID.py` script as follows:

`python -m eval.compute_FID`

This script has the following parameters:
- mean_file1 - Filename which the mean of the ImageNet data is stored.
- mean_file2 - Filename which the mean of the desired model to calculate FID scored for is stored.
- var_file1 - Filename which the variance of the ImageNet data is stored.
- var_file2 - Filename which the variance of the desired model to calculate FID scored for is stored.

Once the script is run, the FID will be printed to the screen.

<b>Note</b>: I have computed the FID for all the pretrained models, which can be found in the same location as [Downloading Pre-Trained Models](#downloading-pre-trained-models) int the Google Drive folder in the filename `saved_stats.7z`. You can use 7-zip to open this file.


# My Results

As stated in [Downloading Pre-Trained Models](#downloading-pre-trained-models), there are 5 different models I tried out:
- res ➜ conv ➜ clsAtn ➜ chnAtn (Res-Conv)
- res ➜ clsAtn ➜ chnAtn (Res)
- res ➜ res ➜ clsAtn ➜ chnAtn (Res-Res)
- res ➜ res ➜ clsAtn ➜ atn ➜ chnAtn (Res-Res-Atn)
- res ➜ clsAtn ➜ chnAtn with 192 channels (Res Large)

Although I trained with classifier-free guidance, I calculated FID scores without guidance as adding guidance requires me to test too many parameters. Additionally, I only collected 10,000 generated images to calculate my FID scores as that already took long enough to generate.

By the way, long FID generation times are one of the problems with diffusion, generation times take forever and unlike GANs, you are not generating images during training. So, you can’t continuously collect FID scores as the model is learning.

Although I keep the classifier guidance value constant, I wanted to test variations between DDIM and DDPM, so I took a look at the step size and the DDIM scale. Note that a DDIM scale of 1 means DDPM, and a scale of 0 means DDIM. A step size of 1 means use all 1000 steps to generate images and a step size of 10 means use 100 steps to generate images:
- DDIM scale 1, step size 1
- DDIM scale 1, step size 10
- DDIM scale 0, step size 1
- DDIM scale 0, step size 10

Let's checkout the FIDs for each of these models:

<p align="center">
  <img src="https://github.com/gmongaras/Diffusion_models_from_scratch/blob/main/results/FID/Res-Conv_FID.png" width="700">

  <img src="https://github.com/gmongaras/Diffusion_models_from_scratch/blob/main/results/FID/Res-Res_FID.png" width="700">

  <img src="https://github.com/gmongaras/Diffusion_models_from_scratch/blob/main/results/FID/Res_FID.png" width="700">

  <img src="https://github.com/gmongaras/Diffusion_models_from_scratch/blob/main/results/FID/Res-Large_FID.png" width="700">

  <img src="https://github.com/gmongaras/Diffusion_models_from_scratch/blob/main/results/FID/Res-Res-Atn_FID.png" width="700">
</p>

It's a little hard to look at in this form. Let's look at a reduced graph with the minimum FID for each model type and u-net construction.

<p align="center">
  <img src="https://github.com/gmongaras/Diffusion_models_from_scratch/blob/main/results/FID/FID_All.png" width="700">
</p>

I calculate the FID score every 50,000 steps. I am only showing the minimum FID score over all 600,000 steps to reduce clutter.

Clearly, the models with two residual blocks performed the best. As for the attention addition, it doesn’t look like it made much of a difference as it was about the same as the model without attention.

Also, using a DDIM (0 scale) with a step size of 10 outperformed all other DDPM/DDIM methods of generation. I find this fact interesting since the model was explicitly trained for DDPM (1 scale) generation on 1000 steps, but performs between with DDIM on 100 steps.

Let's see some sample images using a DDIM scale of 0, classifier-free guidance scale of 4 and classes sampled randomly from the list of classes:

<p align="center">
  <img src="https://github.com/gmongaras/Diffusion_models_from_scratch/blob/main/results/comb.gif" width="500">
  
  <img src="https://github.com/gmongaras/Diffusion_models_from_scratch/blob/main/results/comb.png" width="500">
</p>

Overall, the results look pretty good, though if I trained it for longer and tried to find better hyperparameters, the results could be better!






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
