# Change current directory to the correct directory
import os
# os.chdir('./StReNN')
print(f"Current directory: {os.getcwd()}")

# Import libraries
import numpy as np

import tensorflow as tf
from tensorflow import keras as ks

import IPPy.NN_models as NN_models
from IPPy.metrics import *
from IPPy.utils import *
from IPPy.operators import *
from IPPy.NN_utils import *
from IPPy.GCV_tik import *
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
k_size = 7
sigma = 1.3
kernel = get_motion_blur_kernel(k_size)

kernel_type = 'motion'

noise_level = 0.01

if noise_level == 0:
    suffix = '0'
elif noise_level == 0.025:
    suffix = '025'
elif noise_level == 0.01:
    suffix = '01'

print(f"Suffix: {suffix}")

# Number of epochs
n_epochs = 50

## ----------------------------------------------------------------------------------------------
## ---------- NN --------------------------------------------------------------------------------
## ----------------------------------------------------------------------------------------------
train_nn = False
if train_nn:
    # Define dataloader
    trainloader = Data2D(TRAIN_PATH, kernel, noise_level=noise_level, batch_size=8)

    # Build model and compile it
    model = NN_models.get_UNet(input_shape = (256, 256, 1), n_scales = 4, conv_per_scale = 2, final_relu=True, skip_connection=False)

    # Define the Optimizer
    learning_rate = 1e-3

    model.compile(optimizer=ks.optimizers.Adam(learning_rate=learning_rate),
                loss='mse',
                metrics=[SSIM, 'mse'])

    # Train
    model.fit(trainloader, epochs=n_epochs)
    model.save(f"./model_weights/nn_unet_{suffix}_{kernel_type}.h5")
    print(f"Training of NN model -> Finished.")

## ----------------------------------------------------------------------------------------------
## ---------- Gauss Filter ----------------------------------------------------------------------
## ----------------------------------------------------------------------------------------------
train_stnn = False
if train_stnn:
    # Define dataloader
    sigma = 1
    phi = stabilizers.GaussianFilter(sigma)
    trainloader = Data2D(TRAIN_PATH, kernel, noise_level=noise_level, batch_size=16, phi=phi)

    # Build model and compile it
    model = NN_models.get_UNet(input_shape = (256, 256, 1), n_scales = 4, conv_per_scale = 2, final_relu=True, skip_connection=False)

    # Define the Optimizer
    learning_rate = 1e-3

    model.compile(optimizer=ks.optimizers.Adam(learning_rate=learning_rate),
                loss='mse',
                metrics=[SSIM, 'mse'])

    # Train
    model.fit(trainloader, epochs=n_epochs)
    model.save(f"./model_weights/gauss_unet_{suffix}_{kernel_type}.h5")
    print(f"Training of StNN model -> Finished.")

## ----------------------------------------------------------------------------------------------
## ---------- StNN ------------------------------------------------------------------------------
## ----------------------------------------------------------------------------------------------
train_stnn = True
if train_stnn:
    # Define dataloader
    reg_param = 0.5
    phi = stabilizers.Tik_CGLS_stabilizer(kernel, reg_param, k=10)
    trainloader = Data2D(TRAIN_PATH, kernel, noise_level=noise_level, batch_size=16, phi=phi)

    # Build model and compile it
    model = NN_models.get_UNet(input_shape = (256, 256, 1), n_scales = 4, conv_per_scale = 2, final_relu=True, skip_connection=False)

    # Define the Optimizer
    learning_rate = 1e-3

    model.compile(optimizer=ks.optimizers.Adam(learning_rate=learning_rate),
                loss='mse',
                metrics=[SSIM, 'mse'])

    # Train
    model.fit(trainloader, epochs=n_epochs)
    model.save(f"./model_weights/tik_unet_{suffix}_{kernel_type}.h5")
    print(f"Training of StNN model -> Finished.")