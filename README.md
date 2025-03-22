# Image-Classification-in-TinyML

The CIFAR-10 Dataset is a vital image classification dataset. It consists of 60000 32x32 color images in 10 classes (airplanes, automobiles, birds, cats, deer, dogs, frogs, horses, ships, and trucks), with 6000 images per class. There are 50000 training images and 10000 test images.

## Cifar-10 Dataset
The dataset is divided into five training batches and one test batch, each with 10000 images. The test batch contains exactly 1000 randomly selected images from each class. The training batches contain the remaining images in random order, but some may contain more images from one class than another. Between them, the training batches contain exactly 5000 images from each class.

![cifar10](https://github.com/picoders1/Image-Classification-in-TinyML/assets/87698874/317d59ff-cfab-41e6-ba9c-b258f69f8ae3)

## The GOALS of this project are to:
Implement different Convolutional Neural Network (CNN) classifiers using GPU-enabled Tensorflow and Keras API Compare different CNN architectures.

## Tools:
1. GPU-enabled Tensorflow
2. Keras API

Here are the classes in the dataset, as well as 10 random images from each: 

## Results
A validation dataset of size 10,000 was deduced from the Training dataset, and its size was changed to 40,000. We train the following models for 50 epochs.

Parameters Initialization
Both models have been initialized with random weights sampled from a normal distribution and bias with 0.
These parameters have been initialized only for the Linear layers present in both of the models.
If n represents several nodes in a Linear Layer, then weights are given as a sample of normal distribution in the range (0,y). Here y represents the standard deviation calculated as y=1.0/sqrt(n)
The normal distribution is chosen since the probability of choosing a set of weights closer to zero in the distribution is more than that of the higher values. Unlike in Uniform distribution where the probability of choosing any value is equal.

## Model - 1: FFNN

This Linear Model uses 3072 nodes at the input layer, 2048, 1024, 512, and 256 nodes in the first, second, third, and fourth hidden layers respectively, with an output layer of 10 nodes (10 classes).
The test accuracy is 52.81% (This result uses a dropout probability of 25%)
An FNet_model.pth file has been included. With this one can directly load the model state_dict and use it for testing.

## Model - 2: CNN

The Convolutional Neural Network has 4 convolution layers and pooling layers with 2 fully connected layers. The first convolution layer takes in a channel of dimension 3 since the images are RGB. The kernel size is chosen to be of size 3x3 with a stride of 1. The output of this convolution is set to 16 channels which means it will extract 16 feature maps using 16 kernels. We pad the image with a padding size of 1 so that the input and output dimensions are the same. The output dimension at this layer will be 16 x 32 x 32. We apply RelU activation to it followed by a max-pooling layer with a kernel size of 2 and stride 2. This down-samples the feature maps to dimensions of 16 x 16 x 16.

The second convolution layer will have an input channel size of 16. We choose an output channel size to be 32 which means it will extract 32 feature maps. The kernel size for this layer is 3 with stride 1. We again use a padding size of 1 so that the input and output dimensions remain the same. The output dimension at this layer will be 32 x 16 x 16. We then follow up it with a RelU activation and a max-pooling layer with a kernel of size 2 and stride 2. This down-samples the feature maps to dimensions of 32 x 8 x 8.

The third convolution layer will have an input channel size of 32. We choose an output channel size to be 64 which means it will extract 64 feature maps. The kernel size for this layer is 3 with stride 1. We again use a padding size of 1 so that the input and output dimensions remain the same. The output dimension at this layer will be 64 x 8 x 8. We then follow up it with a RelU activation and a max-pooling layer with a kernel of size 2 and stride 2. This down-samples the feature maps to dimensions of 64 x 4 x 4.

The fourth convolution layer will have an input channel size of 64. We choose an output channel size to be 128 which means it will extract 128 feature maps. The kernel size for this layer is 3 with stride 1. We again use a padding size of 1 so that the input and output dimensions remain the same. The output dimension at this layer will be 128 x 4 x 4 followed up with a RelU activation and a max-pooling layer with a kernel of size 2 and stride 2. This down-samples the feature maps to dimensions of 128 x 2 x 2.

Finally, 3 fully connected layers are used. We will pass a flattened version of the feature maps to the first fully connected layer. The fully connected layers have 512 nodes at the input layer, and 256, and 64 nodes in the first and second hidden layers respectively, with an output layer of 10 nodes (10 classes). So we have two fully connected layers of size 512 x 256 followed up by 256 x 64 and 64 x 10.

![cifar loss curve](https://github.com/picoders1/Image-Classification-in-TinyML/assets/87698874/1a166349-4a65-4a3b-aac1-1236039d8cea)

The test accuracy is 79.85% (This result uses a dropout probability of 25%)

A convNet_model.pth file has been included. With this one can directly load the model state_dict and use it for testing. 


## Thank You!!

