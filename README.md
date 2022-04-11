# NN_Scaling

Here I am training several neural network architectures and recording the minimum loss achieved as a function of varying 1) the model size and 2) the dataset size.

# Literature

* [A Constructive Prediction of the Generalizaiton Error Across Scales](https://arxiv.org/abs/1909.12673)
* [Explaining Neural Scaling Laws](https://arxiv.org/abs/2102.06701)

# Data

MNIST and CIFAR10

# Models

`ResNet50`, `WideResNet50`, `Inception_V3`, `DenseNet`, `VGG16_bn`. For data scaling, models are downloaded pretrained from [torchvision](https://pytorch.org/vision/0.8/models.html). For model scaling, models are *not* pretrained because the torchvision base architecture is modified by scaling the width in a linear layer/the size of kernel in convolutional channels. None of the intiial weights from pretraining would translate over nicely.

# Code

## Notebooks

* Notebooks are kind of a mess, will eventually be cleaned up. Many of them are work-zones; some are used for plotting.

## Scripts

* Networks are written using `PyTorch` and `PyTorch Lightning` for training. Right now they may not be written in the most efficient way, will eventually be cleaned up and optimized (hopefully).

# Notes
* When running must use the syntax python <script.py> <# of epochs> <scaling parameter>
* To run on gpu, the `gpus = -1` flag in the `pl.Trainer` must be included.

