import numpy as np
import matplotlib.pyplot as plt
from skimage import data

from tensorflow import keras as ks

from IPPy.nn import models as NN_models
from IPPy.nn.datasets import *
from IPPy import reconstructors
from IPPy.metrics import *
from IPPy.utils import *
from IPPy.operators import *
from IPPy import stabilizers

import utilities
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
corr_data = np.zeros_like(test_data)
for i in range(len(test_data)):
    x_true = test_data[i]

    y = K @ x_true
    y_delta = y.reshape((m, n)) + noise_level * np.random.normal(0, 1, (m, n)) + delta * np.random.normal(0, 1, (m, n))

    # Poisson
    if peak != 0:
        y_delta = np.random.poisson(y_delta * peak) / peak

    # Add to data
    corr_data[i] = y_delta

PSNR_errors = np.zeros((len(test_data), len(args.model+1)))
SSIM_errors = np.zeros((len(test_data), len(args.model+1)))
for i, recon_name in enumerate(args.model):
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

    rec_data = Psi(corr_data)

    for j in range(len(test_data)):
        PSNR_errors[j+1, i] = PSNR(test_data[j], rec_data[j])
        SSIM_errors[j+1, i] = SSIM(test_data[j], rec_data[j]).numpy()
for j in range(len(test_data)):
        PSNR_errors[j+1, 0] = PSNR(test_data[j], corr_data[j])
        SSIM_errors[j+1, 0] = SSIM(test_data[j], corr_data[j]).numpy()

# Compute statistics
PSNR_means = np.mean(PSNR_errors, axis=0)
PSNR_stds = np.std(PSNR_errors, axis=0)

SSIM_means = np.mean(SSIM_errors, axis=0)
SSIM_stds = np.std(SSIM_errors, axis=0)

############################## TO DO!!
# Print out the results
import tabulate
data = [["", "PSNR", "SSIM"],
        ["Start", str(PSNR_means[0])[:5] + u" \u00B1 " + str(PSNR_stds[0])[:5], str(SSIM_means[0])[:6] + u" \u00B1 " + str(SSIM_stds[0])[:6]],
        ["NN", str(PSNR_means[1])[:5] + u" \u00B1 " + str(PSNR_stds[1])[:5], str(SSIM_means[1])[:6] + u" \u00B1 " + str(SSIM_stds[1])[:6]],
        ["Gauss", str(PSNR_means[2])[:5] + u" \u00B1 " + str(PSNR_stds[2])[:5], str(SSIM_means[2])[:6] + u" \u00B1 " + str(SSIM_stds[2])[:6]],
        ["Tik", str(PSNR_means[3])[:5] + u" \u00B1 " + str(PSNR_stds[3])[:5], str(SSIM_means[3])[:6] + u" \u00B1 " + str(SSIM_stds[3])[:6]]]
print(tabulate.tabulate(data))

# Save the errors
np.save(f'./results/PSNR_{model_type}_{suffix}_{out_domain_label}{kernel_type}.npy', PSNR_errors)
np.save(f'./results/SSIM_{model_type}_{suffix}_{out_domain_label}{kernel_type}.npy', SSIM_errors)

