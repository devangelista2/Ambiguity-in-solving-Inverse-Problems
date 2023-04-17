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
k_size = 11
sigma = 1.3

kernel_type = 'gaussian'
if kernel_type == 'gaussian':
    kernel = get_gaussian_kernel(k_size, sigma)
elif kernel_type == 'motion':
    kernel = get_motion_blur_kernel(k_size) 

model_type = 'unet' # choose unet, ssnet, baseline, nafnet

noise_level = 0.025
delta = 0.050 # Out-of-domain noise intensity
peak = 0 # Peak in Poisson Noise

if noise_level == 0:
    suffix = '0'
elif noise_level == 0.025:
    suffix = '025'

if delta != 0:
    out_domain_label = "g_noise_"
elif delta == 0 and peak != 0:
    out_domain_label = "poisson_noise_"
else:
    out_domain_label = ""

print(f"Suffix: {suffix}")

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

## ----------------------------------------------------------------------------------------------
## ---------- NN --------------------------------------------------------------------------------
## ----------------------------------------------------------------------------------------------
test_nn = True
if test_nn:
    recon_name = 'nn'
    weights_name = f'{recon_name}_{model_type}'
    phi = stabilizers.PhiIdentity()

    model = utilities.load_model_by_parameters(recon_name, model_type,
                                               suffix, kernel_type)
    Psi = reconstructors.StabilizedReconstructor(model, phi)

    nn_data = Psi(corr_data)
    print(nn_data.shape)

## ----------------------------------------------------------------------------------------------
## ---------- Gauss Filter ----------------------------------------------------------------------
## ----------------------------------------------------------------------------------------------
test_gauss = True
if test_gauss:
    # Reconstruct with FiNN
    recon_name = 'gauss'
    weights_name = f'{recon_name}_{model_type}'
    sigma = 1

    phi = stabilizers.GaussianFilter(sigma)
    model = utilities.load_model_by_parameters(recon_name, model_type,
                                               suffix, kernel_type)
    Psi = reconstructors.StabilizedReconstructor(model, phi)

    gauss_data = Psi(corr_data)

## ----------------------------------------------------------------------------------------------
## ---------- StNN ------------------------------------------------------------------------------
## ----------------------------------------------------------------------------------------------
test_stnn = True
if test_stnn:
    # Reconstruct with StNN
    recon_name = 'tik'
    weights_name = f'{recon_name}_{model_type}'
    if noise_level == 0:
        reg_param = 1e-2
    elif noise_level == 0.025:
        reg_param = 1e-3
    phi = stabilizers.Tik_CGLS_stabilizer(kernel, reg_param, k=2)

    model = utilities.load_model_by_parameters(recon_name, model_type,
                                               suffix, kernel_type)
    Psi = reconstructors.StabilizedReconstructor(model, phi)

    tik_data = Psi(corr_data)

# Compute the avg and std of the errors, in PSNR and SSIM.
PSNR_errors = np.zeros((len(test_data), 4))
SSIM_errors = np.zeros((len(test_data), 4))
for i in range(len(test_data)):
    PSNR_errors[i, 0] = PSNR(test_data[i], corr_data[i])
    PSNR_errors[i, 1] = PSNR(test_data[i], nn_data[i])
    PSNR_errors[i, 2] = PSNR(test_data[i], gauss_data[i])
    PSNR_errors[i, 3] = PSNR(test_data[i], tik_data[i])

    SSIM_errors[i, 0] = SSIM(test_data[i], corr_data[i]).numpy()
    SSIM_errors[i, 1] = SSIM(test_data[i], nn_data[i]).numpy()
    SSIM_errors[i, 2] = SSIM(test_data[i], gauss_data[i]).numpy()
    SSIM_errors[i, 3] = SSIM(test_data[i], tik_data[i]).numpy()

PSNR_means = np.mean(PSNR_errors, axis=0)
PSNR_stds = np.std(PSNR_errors, axis=0)

SSIM_means = np.mean(SSIM_errors, axis=0)
SSIM_stds = np.std(SSIM_errors, axis=0)

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

