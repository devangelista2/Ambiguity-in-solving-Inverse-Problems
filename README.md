# Ambiguity-in-solving-Inverse-Problems
GitHub repository to reproduce experiments from the paper: ...

## Installation
The use the code, simply clone the GitHub repository locally with

```
git clone https://github.com/devangelista2/UnderstandingStabilizers.git
```

Moreover, the `IPPy` library is required to execute portions of the code. Please refer to `IPPy` documentation for an explanation on how to install it.

## Project Stucture
Please note that the functions requires a specific folders and files structure to work. Since, due to memory constraint, it was not possible to upload the whole project on GitHub, the user is asked to create some folders to follow the required structure.  This can be obtained by simply creating the `data` and the `model_weights` folders by running:

```
mkdir data
mkdir gaussian_blur/model_weights
mkdir motion_blur/model_weights
```

Into the main project folder. For informations about how to download the data (to be placed inside the `data` folder), and the pre-trained model weights, please refer to the following.

## Datasets
To run the experiments, the training and the test set has to be downloaded. A copy of the data used to train the models and get the results for the paper is available on HuggingFace. To get it, simply create a folder named `data` into the main project directory, move into that and run the following command:

```
git lfs install
git clone https://huggingface.co/datasets/TivoGatto/celeba_grayscale
```

which will download the data, in `.npy` format, used in the experiments. It is a slighly modified version of the GoPro dataset, where the images has been cropped to be $256 \times 256$ images, and converted to grey-scale with standard conversion algorithms. 

## Pre-trained models

The pre-trained models will be available soon.
