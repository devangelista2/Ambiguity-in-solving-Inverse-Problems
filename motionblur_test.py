import numpy as np
import matplotlib.pyplot as plt
from skimage import data

from tensorflow import keras as ks

import IPPy.NN_models as NN_models
from IPPy import reconstructors
from IPPy.metrics import *
from IPPy.utils import *
from IPPy.operators import *
from IPPy.NN_utils import *
from IPPy import stabilizers


## ----------------------------------------------------------------------------------------------
## ---------- Initialization --------------------------------------------------------------------
## ----------------------------------------------------------------------------------------------

# Load data
DATA_PATH = '../data/'
TRAIN_PATH = os.path.join(DATA_PATH, 'GOPRO_train_small.npy')
TEST_PATH = os.path.join(DATA_PATH, 'GOPRO_test_small.npy')

train_data = np.load(TRAIN_PATH)
test_data = np.load(TEST_PATH)
N_train, m, n = train_data.shape
print(f"Training data shape: {train_data.shape}")

# Define the setup for the forward problem
k_size = 7
sigma = 1.3

kernel_type = 'motion'
if kernel_type == 'gaussian':
    kernel = get_gaussian_kernel(k_size, sigma)
elif kernel_type == 'motion':
    kernel = get_motion_blur_kernel(k_size) 

noise_level = 0.01
delta = 0 # Out-of-domain noise intensity
peak = 0 # Peak in Poisson Noise

if noise_level == 0:
    suffix = '0'
elif noise_level == 0.01:
    suffix = '01'

if delta != 0:
    out_domain_label = "g_noise_"
elif delta == 0 and peak != 0:
    out_domain_label = "poisson_noise_"
else:
    out_domain_label = ""

print(f"Suffix: {suffix}")

# Corrupt
x_true = test_data[557]
K = ConvolutionOperator(kernel, (m, n))
y = K @ x_true
y_delta = y.reshape((m, n))


# Poisson
if peak != 0:
    y_delta = np.random.poisson(y_delta * peak) / peak

# Gauss
y_delta = y_delta + noise_level * np.random.normal(0, 1, (m, n)) + delta * np.random.normal(0, 1, (m, n)) 

# Save the results
plt.imsave(f'./results/true_{suffix}_{out_domain_label}{kernel_type}.png', x_true.reshape((m, n)), cmap='gray', dpi=400)
plt.imsave(f'./results/corr_{suffix}_{out_domain_label}{kernel_type}.png', y_delta.reshape((m, n)), cmap='gray', dpi=400)

## ----------------------------------------------------------------------------------------------
## ---------- NN --------------------------------------------------------------------------------
## ----------------------------------------------------------------------------------------------
test_nn = True
if test_nn:
    weights_name = 'nn_unet'
    phi = stabilizers.PhiIdentity()

    model = ks.models.load_model(f"./model_weights/{weights_name}_{suffix}_{kernel_type}.h5", custom_objects={'SSIM': SSIM})
    Psi = reconstructors.StabilizedReconstructor(model, phi)

    x_nn = Psi(y_delta)

    # Save the results
    plt.imsave(f'./results/{weights_name}_{suffix}_{out_domain_label}{kernel_type}_recon.png', x_nn, cmap='gray', dpi=400)

## ----------------------------------------------------------------------------------------------
## ---------- Gauss Filter ----------------------------------------------------------------------
## ----------------------------------------------------------------------------------------------
test_gauss = True
if test_gauss:
    # Reconstruct with StNN
    weights_name = 'gauss_unet'
    sigma = 1
    phi = stabilizers.GaussianFilter(sigma)

    # Save the results
    plt.imsave(f'./results/{weights_name}_{suffix}_{out_domain_label}{kernel_type}_preprocess.png', phi(y_delta), cmap='gray', dpi=400)

    model = ks.models.load_model(f"./model_weights/{weights_name}_{suffix}_{kernel_type}.h5", custom_objects={'SSIM': SSIM})
    Psi = reconstructors.StabilizedReconstructor(model, phi)

    x_gauss = Psi(y_delta)

    # Save the results
    plt.imsave(f'./results/{weights_name}_{suffix}_{out_domain_label}{kernel_type}_recon.png', x_gauss, cmap='gray', dpi=400)

## ----------------------------------------------------------------------------------------------
## ---------- StNN ------------------------------------------------------------------------------
## ----------------------------------------------------------------------------------------------
test_stnn = True
if test_stnn:
    # Reconstruct with StNN
    weights_name = 'tik_unet'
    if noise_level == 0:
        reg_param = 1
    elif noise_level == 0.01:
        reg_param = 0.5
    phi = stabilizers.Tik_CGLS_stabilizer(kernel, reg_param, k=10)

    # Save the results
    plt.imsave(f'./results/{weights_name}_{suffix}_{out_domain_label}{kernel_type}_preprocess.png', phi(y_delta), cmap='gray', dpi=400)

    model = ks.models.load_model(f"./model_weights/{weights_name}_{suffix}_{kernel_type}.h5", custom_objects={'SSIM': SSIM})
    Psi = reconstructors.StabilizedReconstructor(model, phi)

    x_tik = Psi(y_delta)

    # Save the results
    plt.imsave(f'./results/{weights_name}_{suffix}_{out_domain_label}{kernel_type}_recon.png', x_tik, cmap='gray', dpi=400)

# Quanitative results
if test_nn and test_gauss and test_stnn:
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