import numpy as np
import matplotlib.pyplot as plt
from skimage import data

import utilities

from tensorflow import keras as ks

import IPPy.nn.models as NN_models
from IPPy import reconstructors
from IPPy.metrics import *
from IPPy.utils import *
from IPPy.operators import *
from IPPy.nn.datasets import *
from IPPy import stabilizers

import os
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
parser.add_argument("-dm", "--delta_min",
                    help="Minimum noise level of additional corruption. Given as gaussian variance. Default: 0.01.",
                    type=float,
                    required=False,
                    default=0.01
                    )
parser.add_argument("-dM", "--delta_max",
                    help="Maximum noise level of additional corruption. Given as gaussian variance. Default: 0.1.",
                    type=float,
                    required=False,
                    default=0.1
                    )
parser.add_argument("-dn", "--delta_n",
                    help="Number of noise level of additional corruption. Default: 10.",
                    type=int,
                    required=False,
                    default=10
                    )
parser.add_argument('--config',
                    help="The path for the .yml containing the configuration for the model.",
                    type=str,
                    required=False,
                    default=None)
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
DATA_PATH = './data/'
TRAIN_PATH = os.path.join(DATA_PATH, 'GOPRO_train_small.npy')
TEST_PATH = os.path.join(DATA_PATH, 'GOPRO_test_small.npy')

train_data = np.load(TRAIN_PATH)
test_data = np.load(TEST_PATH)
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

# Corrupt
x_true = test_data[0]
K = ConvolutionOperator(kernel, (m, n))
y = K @ x_true
y = y.reshape((m, n))

errors = np.zeros((3, args.delta_n))
for j, recon_name in enumerate(args.model):
    for i, d in enumerate(np.linspace(args.delta_min, args.delta_max, args.delta_n)):
        y_delta = y + d * np.random.normal(0, 1, y.shape)

        ## ----------------------------------------------------------------------------------------------
        ## ---------- NN --------------------------------------------------------------------------------
        ## ----------------------------------------------------------------------------------------------
        if recon_name == 'nn':
            weights_name = f'{recon_name}_{args.model_type}'
            phi = stabilizers.PhiIdentity()

        ## ----------------------------------------------------------------------------------------------
        ## ---------- Gauss Filter ----------------------------------------------------------------------
        ## ----------------------------------------------------------------------------------------------
        if recon_name == 'finn':
            # Reconstruct with FiNN
            weights_name = f'{recon_name}_{args.model_type}'
            sigma = setup[recon_name]['sigma']
            phi = stabilizers.GaussianFilter(sigma)

        ## ----------------------------------------------------------------------------------------------
        ## ---------- StNN ------------------------------------------------------------------------------
        ## ----------------------------------------------------------------------------------------------
        if recon_name == 'stnn':
            # Reconstruct with StNN
            weights_name = f'{recon_name}_{args.model_type}'
            reg_param = setup[recon_name]['reg_param']
            phi = stabilizers.Tik_CGLS_stabilizer(kernel, reg_param, k=setup[recon_name]['n_iter'])

        model = utilities.load_model_by_parameters(recon_name, args.model_type,
                                                suffix, kernel_type)
        Psi = reconstructors.StabilizedReconstructor(model, phi)

        x_rec = Psi(y_delta)

        # Append errors
        errors[j, i] = np.linalg.norm(x_rec.flatten() - x_true.flatten()) / np.linalg.norm(x_true.flatten())
        print(f"Iteration {i+1}.")
np.save(f'relerrors_ssnet_{suffix}_{args.delta_max}.npy', errors)