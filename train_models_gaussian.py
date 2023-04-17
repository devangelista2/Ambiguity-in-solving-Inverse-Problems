# Change current directory to the correct directory
import os
# os.chdir('./StReNN')
print(f"Current directory: {os.getcwd()}")

# Import libraries
import numpy as np

import tensorflow as tf
from tensorflow import keras as ks

import IPPy.nn.models as models
from IPPy.metrics import *
from IPPy.utils import *
from IPPy.operators import *
from IPPy.nn.datasets import *
from IPPy import stabilizers

## ----------------------------------------------------------------------------------------------
## ---------- Initialization --------------------------------------------------------------------
## ----------------------------------------------------------------------------------------------

# Load data
DATA_PATH = './data/'
TRAIN_PATH = os.path.join(DATA_PATH, 'GOPRO_train_small.npy')

train_data = np.load(TRAIN_PATH)
N_train, m, n = train_data.shape
print(f"Training data shape: {train_data.shape}")

# Define the setup for the forward problem
k_size = 11
sigma = 1.3
kernel = get_gaussian_kernel(k_size, sigma)

kernel_type = 'gaussian'

noise_level = 0.025

if noise_level == 0:
    suffix = '0'
elif noise_level == 0.025:
    suffix = '025'
elif noise_level == 0.01:
    suffix = '01'

print(f"Suffix: {suffix}")

# Number of epochs
n_epochs = 50
batch_size = 8

# Choose between 'ssnet' or 'unet' or 'baseline' or 'nafnet'
model_type = 'nafnet'

## ----------------------------------------------------------------------------------------------
## ---------- NN --------------------------------------------------------------------------------
## ----------------------------------------------------------------------------------------------
train_nn = True
if train_nn:
    # Define dataloader
    trainloader = Data2D(TRAIN_PATH, kernel, noise_level=noise_level, batch_size=batch_size)

    # Build model and compile it
    if model_type == 'unet':
        model = models.get_UNet(input_shape = (256, 256, 1), n_scales = 4, conv_per_scale = 2, final_relu=True, skip_connection=False)
    elif model_type == 'ssnet':
        model = models.get_SSNet(input_shape = (256, 256, 1), n_ch=(128, 128, 128), k_size=(9, 5, 3), final_relu=True, skip_connection=False)
    elif model_type == 'baseline':
        model = models.get_BaselineModel(input_shape = (256, 256, 1), conv_per_scale=[3, 3, 3], init_conv=32)
    elif model_type == 'nafnet':
        model = models.get_NAFModel(input_shape = (256, 256, 1), conv_per_scale=[1, 1, 1, 28], init_conv=32)
        

    # We follow the NAFNet optimization scheme.
    learning_rate_schedule = tf.keras.optimizers.schedules.CosineDecay(1e-3, 13_000, alpha=1e-6)
    optimizer = ks.optimizers.Adam(learning_rate=learning_rate_schedule, beta_1=0.9, beta_2=0.9)

    def psnr_loss(y_true, y_pred):
        return tf.reduce_mean(tf.math.log(tf.reduce_mean((y_true - y_pred) ** 2, axis=(1, 2, 3))) + 1e-8)

    # Define the Optimizer
    learning_rate = 1e-3

    model.compile(optimizer=optimizer,
                loss=psnr_loss,
                metrics=[SSIM, 'mse'])

    # Train
    model.fit(trainloader, epochs=n_epochs)
    model.save(f"./model_weights/nn_{model_type}_{suffix}_{kernel_type}.h5")
    print(f"Training of NN model -> Finished.")

## ----------------------------------------------------------------------------------------------
## ---------- Gauss Filter ----------------------------------------------------------------------
## ----------------------------------------------------------------------------------------------
train_stnn = False
if train_stnn:
    # Define dataloader
    sigma = 1
    phi = stabilizers.GaussianFilter(sigma)
    trainloader = Data2D(TRAIN_PATH, kernel, noise_level=noise_level, batch_size=batch_size, phi=phi)

    # Build model and compile it
    if model_type == 'unet':
        model = models.get_UNet(input_shape = (256, 256, 1), n_scales = 4, conv_per_scale = 2, final_relu=True, skip_connection=False)
    elif model_type == 'ssnet':
        model = models.get_SSNet(input_shape = (256, 256, 1), n_ch=(128, 128, 128), k_size=(9, 5, 3), final_relu=True, skip_connection=False)
    elif model_type == 'baseline':
        model = models.get_BaselineModel(input_shape = (256, 256, 1), conv_per_scale=[3, 3, 3], init_conv=32)
    elif model_type == 'nafnet':
        model = models.get_NAFModel(input_shape = (256, 256, 1), conv_per_scale=[3, 3, 3], init_conv=32)

    # Define the Optimizer
    learning_rate = 1e-3

    model.compile(optimizer=ks.optimizers.Adam(learning_rate=learning_rate),
                loss='mse',
                metrics=[SSIM, 'mse'])

    # Train
    model.fit(trainloader, epochs=n_epochs)
    model.save(f"./model_weights/gauss_{model_type}_{suffix}_{kernel_type}.h5")
    print(f"Training of StNN model -> Finished.")

## ----------------------------------------------------------------------------------------------
## ---------- StNN ------------------------------------------------------------------------------
## ----------------------------------------------------------------------------------------------
train_stnn = False
if train_stnn:
    # Define dataloader
    reg_param = 1e-2
    phi = stabilizers.Tik_CGLS_stabilizer(kernel, reg_param, k=3)
    trainloader = Data2D(TRAIN_PATH, kernel, noise_level=noise_level, batch_size=batch_size, phi=phi)

    # Build model and compile it
    if model_type == 'unet':
        model = models.get_UNet(input_shape = (256, 256, 1), n_scales = 4, conv_per_scale = 2, final_relu=True, skip_connection=False)
    elif model_type == 'ssnet':
        model = models.get_SSNet(input_shape = (256, 256, 1), n_ch=(128, 128, 128), k_size=(9, 5, 3), final_relu=True, skip_connection=False)
    elif model_type == 'baseline':
        model = models.get_BaselineModel(input_shape = (256, 256, 1), conv_per_scale=[3, 3, 3], init_conv=32)
    elif model_type == 'nafnet':
        model = models.get_NAFModel(input_shape = (256, 256, 1), conv_per_scale=[3, 3, 3], init_conv=32)

    # Define the Optimizer
    learning_rate = 1e-3

    model.compile(optimizer=ks.optimizers.Adam(learning_rate=learning_rate),
                loss='mse',
                metrics=[SSIM, 'mse'])

    # Train
    model.fit(trainloader, epochs=n_epochs)
    model.save(f"./model_weights/tik_{model_type}_{suffix}_{kernel_type}.h5")
    print(f"Training of StNN model -> Finished.")