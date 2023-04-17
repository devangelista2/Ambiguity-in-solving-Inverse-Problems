import numpy as np
import matplotlib.pyplot as plt
from skimage import data, transform

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
DATA_PATH = './custom_image_results/elephant.jpg'
print(f"Image: {DATA_PATH}")

x_true = utilities.normalize(plt.imread(DATA_PATH))

m, n, c = x_true.shape
print(f"Image shape: {x_true.shape}")

# Define the setup for the forward problem
k_size = 11
sigma = 1.3

kernel_type = 'gaussian'
kernel = get_gaussian_kernel(k_size, sigma)

model_type = 'nafnet' # choose unet, ssnet, baseline, nafnet

noise_level = 0.025
delta = 0.05 # Out-of-domain noise intensity

if noise_level == 0:
    suffix = '0'
elif noise_level == 0.025:
    suffix = '025'

if delta != 0:
    out_domain_label = "g_noise_"
else:
    out_domain_label = ""

print(f"Suffix: {suffix}")

# Corrupt
K = ConvolutionOperator(kernel, (m, n))

y = x_true.copy()
for i in range(c):
    y[:, :, i] = (K @ x_true[:, :, i]).reshape((m, n))
e = delta * np.random.normal(0, 1, (m, n, c))
y_delta = y + e

# Save the results
plt.imsave(f'./custom_image_results/corr_{suffix}_{out_domain_label}{kernel_type}.jpg', utilities.normalize(y_delta), dpi=400)
plt.imsave(f'./custom_image_results/noise_{suffix}_{out_domain_label}{kernel_type}.jpg', utilities.normalize(e), dpi=400)

# Change the shape of y_delta to work on it
y_delta = np.expand_dims(np.transpose(y_delta, (2, 0, 1)), -1)
print(f"Final shape: {y_delta.shape}")

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

    x_nn = Psi(y_delta)[:, :, :, 0]
    x_nn = np.transpose(x_nn, (1, 2, 0))

    # Save the results
    plt.imsave(f'./custom_image_results/{weights_name}_{suffix}_{out_domain_label}{kernel_type}_recon.jpg', utilities.normalize(x_nn), dpi=400)

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

    # Save the results
    x_gauss_pre = np.transpose(phi(y_delta)[:, :, :, 0], (1, 2, 0))
    plt.imsave(f'./custom_image_results/{weights_name}_{suffix}_{out_domain_label}{kernel_type}_preprocess.png', utilities.normalize(x_gauss_pre), dpi=400)

    model = utilities.load_model_by_parameters(recon_name, model_type,
                                               suffix, kernel_type)
    Psi = reconstructors.StabilizedReconstructor(model, phi)

    x_gauss = Psi(y_delta)[:, :, :, 0]
    x_gauss = np.transpose(x_gauss, (1, 2, 0))

    # Save the results
    plt.imsave(f'./custom_image_results/{weights_name}_{suffix}_{out_domain_label}{kernel_type}_recon.jpg', utilities.normalize(x_gauss), dpi=400)

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
        reg_param = 1e-2
    phi = stabilizers.Tik_CGLS_stabilizer(kernel, reg_param, k=3)

    # Save the results
    x_gauss_pre = np.transpose(np.array([phi(y_delta[i, :, :, 0]) for i in range(c)]), (1, 2, 0))
    plt.imsave(f'./custom_image_results/{weights_name}_{suffix}_{out_domain_label}{kernel_type}_preprocess.png', utilities.normalize(x_gauss_pre), dpi=400)

    model = utilities.load_model_by_parameters(recon_name, model_type,
                                               suffix, kernel_type)
    Psi = reconstructors.StabilizedReconstructor(model, phi)

    x_tik = np.array([Psi(y_delta[i, :, :, 0]) for i in range(c)])[:, 0, :, :, 0]
    x_tik = np.transpose(x_tik, (1, 2, 0))

    # Save the results
    plt.imsave(f'./custom_image_results/{weights_name}_{suffix}_{out_domain_label}{kernel_type}_recon.jpg', utilities.normalize(x_tik), dpi=400)

# Quanitative results
if test_nn and test_gauss and test_stnn:
    # Adjust the shape of y
    y_delta = np.transpose(y_delta[:, :, :, 0], (1, 2, 0))

    # Compute the PSNR
    psnr_start = PSNR(x_true, y_delta)
    psnr_nn = PSNR(x_true, x_nn)
    psnr_gauss = PSNR(x_true, x_gauss)
    psnr_tik = PSNR(x_true, x_tik)

    # Compute the SSIM
    ssim_start = SSIM(x_true, y_delta).numpy()
    ssim_nn = SSIM(x_true, x_nn).numpy()
    ssim_gauss = SSIM(x_true, x_gauss).numpy()
    ssim_tik = SSIM(x_true, x_tik).numpy()

    # Print ressults
    import tabulate
    data = [["", "PSNR", "SSIM"],
            ["Start", psnr_start, ssim_start],
            ["NN", psnr_nn, ssim_nn],
            ["Gauss", psnr_gauss, ssim_gauss],
            ["Tik", psnr_tik, ssim_tik]]

    print(tabulate.tabulate(data))