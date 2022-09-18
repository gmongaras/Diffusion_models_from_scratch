# Diffusion_models_from_scratch
Creating a diffusion model from scratch in PyTorch to learn exactly how they work.


# Todo:
1. Basic PixelCNN++
2. Improve efficiency of convolution with masks (in the code they have parallel models with different convolution masks instead of a single movel witha single convolution mask)
3. RGB pixels values should be produced sequentially, not in parallel. G is conditioned on R, B is conditioned on R and G.



# Data
## ImageNet
https://www.kaggle.com/c/imagenet-object-localization-challenge/overview/description

## Mini-ImageNet
Used for creating the model on my laptop

https://www.kaggle.com/datasets/whitemoon/miniimagenet?select=mini-imagenet-cache-train.pkl

## Cifar-10 and Cifar-100
https://www.cs.toronto.edu/~kriz/cifar.html




# References

Big main paper: https://arxiv.org/abs/2006.11239



Model: https://arxiv.org/abs/2006.11239

https://arxiv.org/abs/2006.11239

Big thanks to: https://bjlkeng.github.io/posts/pixelcnn/