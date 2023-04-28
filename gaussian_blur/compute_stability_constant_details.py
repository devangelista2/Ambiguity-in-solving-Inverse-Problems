import numpy as np
import matplotlib.pyplot as plt
from skimage import data

import os
import utilities

from tensorflow import keras as ks

import IPPy.nn.models as NN_models
from IPPy import reconstructors
from IPPy.metrics import *
from IPPy.utils import *
from IPPy.operators import *
from IPPy.nn.datasets import *
from IPPy import stabilizers

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
parser.add_argument("-d", "--delta",
                    help="Noise level of additional corruption. Given as gaussian variance. Default: 0.",
                    type=float,
                    required=False,
                    default=0
                    )
parser.add_argument("-po", "--poisson",
                    help="Poisson noise level of additional corruption. Given as the Poisson peak. Default: 0.",
                    type=float,
                    required=False,
                    default=0
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

delta = args.delta # Out-of-domain noise intensity
delta_suffix = str(delta).split('.')[-1]

peak = args.poisson # Peak in Poisson Noise

if delta != 0:
    out_domain_label = f"g_noise{delta_suffix}_"
elif delta == 0 and peak != 0:
    out_domain_label = "poisson_noise_"
else:
    out_domain_label = ""

# Corrupt
K = ConvolutionOperator(kernel, (m, n))
blur_data = np.zeros_like(test_data)
corr_data = np.zeros_like(test_data)
for i in range(len(test_data)):
    x_true = test_data[i]

    y = K @ x_true
    y_delta = y.reshape((m, n))

    # Poisson
    if peak != 0:
        y_delta = np.random.poisson(y_delta * peak) / peak

    # Gauss
    e_norm = np.random.random() * (noise_level + delta) * np.sqrt(m*n)
    e = np.random.normal(0, 1, (m, n))
    y_delta = y_delta + e / np.linalg.norm(e.flatten()) * e_norm

    # Add to data
    blur_data[i] = y.reshape((m, n))
    corr_data[i] = y_delta

clean_err = np.zeros((len(test_data), ))
corr_err = np.zeros((len(test_data), ))
norm_e = np.zeros((len(test_data), ))
for recon_name in args.model:
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

    x_rec = Psi(blur_data)
    x_corr_rec = Psi(corr_data)

    # Quanitative results
    clean_err = np.linalg.norm(test_data.reshape((len(test_data), -1)) - x_rec.reshape((len(test_data), -1)), axis=-1)
    corr_err = np.linalg.norm(test_data.reshape((len(test_data), -1)) - x_corr_rec.reshape((len(test_data), -1)), axis=-1)
    norm_e = np.linalg.norm(corr_data.reshape((len(test_data), -1)) - blur_data.reshape((len(test_data), -1)), axis=-1)

    # Save the results
    np.save(f'./gaussian_blur/results/clean_err_{recon_name}_{args.model_type}_{suffix}_{delta_suffix}_gaussian.npy', clean_err)
    np.save(f'./gaussian_blur/results/corr_err_{recon_name}_{args.model_type}_{suffix}_{delta_suffix}_gaussian.npy', corr_err)
    np.save(f'./gaussian_blur/results/norm_e_{recon_name}_{args.model_type}_{suffix}_{delta_suffix}_gaussian.npy', norm_e)

    # Compute the accuracy^{-1}
    inv_acc = np.max(clean_err)

    # Compute the "local stability constant" for each datapoint
    C_local = (corr_err - inv_acc) / norm_e

    # Compute the Global stability constant
    C = np.max(C_local)

    # Print out the result
    print(f"Delta: {delta}. Network: {args.model_type}. Reconstructor: {recon_name}. Acc: {1 / inv_acc}. C: {C}.")