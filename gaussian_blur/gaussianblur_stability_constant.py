import numpy as np
import matplotlib.pyplot as plt
from skimage import data

import os
import utilities

from tensorflow import keras as ks

import IPPy.NN_models as NN_models
from IPPy import reconstructors
from IPPy.metrics import *
from IPPy.utils import *
from IPPy.operators import *
from IPPy.NN_utils import *
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
    y_delta = y_delta + (noise_level + delta) * np.random.normal(0, 1, (m, n))

    # Add to data
    blur_data[i] = y.reshape((m, n))
    corr_data[i] = y_delta

clean_err = []
corr_err = []
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
    err = 0
    corr_e = 0
    for i in range(len(test_data)):
        err += np.linalg.norm(x_true.flatten() - x_rec[i].flatten())
        corr_err += np.linalg.norm(x_true.flatten() - x_corr_rec[i].flatten())
    # Normalize
    err = err / len(test_data)
    corr_e = corr_e / len(test_data)

    # Compute the accuracy
    clean_err.append(err)
    corr_err.append(corr_e)
# Convert to numpy arrays
clean_err = np.array(clean_err)
corr_err = np.array(corr_err)

# Compute the stability
e = y_delta - y
e_norm = np.linalg.norm(e.flatten())

# Print ressults
import tabulate
data = [[""] + args.model,
        ["Acc"] + list(1/clean_err),
        ["C"] + list((corr_err - clean_err) / e_norm),
        ["Err"] + list(clean_err),
        ["Corr_Err"] + list(corr_err)]

print(f"||e|| = {e_norm}.")
print(tabulate.tabulate(data))