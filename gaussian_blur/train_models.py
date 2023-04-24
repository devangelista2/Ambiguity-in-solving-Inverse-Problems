# Import libraries
import numpy as np

import os

import tensorflow as tf
from tensorflow import keras as ks

import IPPy.nn.models as models
from IPPy.metrics import *
from IPPy.utils import *
from IPPy.operators import *
from IPPy.nn.datasets import *
from IPPy import stabilizers

import utilities

import argparse
import yaml

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model",
                    help="Name of the model to process. Can be used for multiple models to compare them.",
                    required=True,
                    action='append',
                    choices=["nn", "finn", "stnn",]
                    )
parser.add_argument('-n', '--model_type',
                    choices=['unet', 'ssnet', 'nafnet'],
                    help='Select the architecture you want to test. Default: unet.',
                    type=str,
                    default='unet',
                    required=False)
parser.add_argument('-ni', '--noise_inj',
                    help="The amount of noise injection. Given as the variance of the Gaussian. Default: 0.",
                    type=float,
                    default=0,
                    required=False)
parser.add_argument('--config',
                    help="The path for the .yml containing the configuration for the model.",
                    type=str,
                    required=False,
                    default=None)
parser.add_argument('--verbose',
                    help="Unable/Disable verbose for the code.",
                    type=str,
                    required=False,
                    default="1",
                    choices=["0", "1"])
args = parser.parse_args()

if args.config is None:
    suffix = str(args.noise_inj).split('.')[-1]
    args.config = f"./config/GoPro_{suffix}_gaussian.yml"

with open(args.config, 'r') as file:
    setup = yaml.safe_load(file)

## ----------------------------------------------------------------------------------------------
## ---------- Initialization --------------------------------------------------------------------
## ----------------------------------------------------------------------------------------------
# Load data
DATA_PATH = 'C:/Users/tivog/data/gopro_small'
TRAIN_PATH = os.path.join(DATA_PATH, 'GOPRO_train_small.npy')

train_data = np.load(TRAIN_PATH)
N_train, m, n = train_data.shape
print(f"Training data shape: {train_data.shape}")

# Define the setup for the forward problem
k_size = setup['k']
sigma = setup['sigma']
kernel = get_gaussian_kernel(k_size, sigma)

kernel_type = 'gaussian'

noise_level = args.noise_inj
suffix = str(noise_level).split('.')[-1]

print(f"Suffix: {suffix}")

# Number of epochs
n_epochs = setup['n_epochs']
batch_size = setup['batch_size']

for recon in args.model:
    ## ----------------------------------------------------------------------------------------------
    ## ---------- NN --------------------------------------------------------------------------------
    ## ----------------------------------------------------------------------------------------------
    if recon == 'nn':
        # Define dataloader
        trainloader = Data2D(TRAIN_PATH, kernel, noise_level=noise_level, batch_size=batch_size)

    ## ----------------------------------------------------------------------------------------------
    ## ---------- FiNN ------------------------------------------------------------------------------
    ## ----------------------------------------------------------------------------------------------
    if recon == 'finn':
        # Define dataloader
        sigma = setup[recon]['sigma']
        phi = stabilizers.GaussianFilter(sigma)
        trainloader = Data2D(TRAIN_PATH, kernel, noise_level=noise_level, batch_size=batch_size, phi=phi)

    ## ----------------------------------------------------------------------------------------------
    ## ---------- StNN ------------------------------------------------------------------------------
    ## ----------------------------------------------------------------------------------------------
    if recon == 'stnn':
        # Define dataloader
        reg_param = setup[recon]['reg_param']
        phi = stabilizers.Tik_CGLS_stabilizer(kernel, reg_param, k=setup[recon]['n_iter'])
        trainloader = Data2D(TRAIN_PATH, kernel, noise_level=noise_level, batch_size=batch_size, phi=phi)

    # Build model and compile it
    model = utilities.build_model(args.model_type)

    # Define the Optimizer
    learning_rate = setup['learning_rate']

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate,
                                                     beta_1=0.9,
                                                     beta_2=0.9),
                  loss=utilities.psnr_loss,
                  metrics=[SSIM, 'mse'])

    #model.compile(optimizer=ks.optimizers.Adam(learning_rate=learning_rate),
    #            loss='mse',
    #            metrics=[SSIM, 'mse'])

    # Train
    model.fit(trainloader, epochs=n_epochs)
    model.save(f"./gaussian_blur/model_weights/{recon}_{args.model_type}_{suffix}_{kernel_type}.h5")
    print(f"Training of {recon} model -> Finished.")