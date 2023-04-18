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

noise_level = 0
delta = 0.01 # Out-of-domain noise intensity
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
y = y.copy().reshape((m, n))


# Poisson
if peak != 0:
    y_delta = np.random.poisson(y_delta * peak) / peak

# Gauss
y_delta = y_delta + (noise_level + delta) * np.random.normal(0, 1, (m, n))


## ----------------------------------------------------------------------------------------------
## ---------- NN --------------------------------------------------------------------------------
## ----------------------------------------------------------------------------------------------
test_nn = True
if test_nn:
    weights_name = 'nn_unet'
    phi = stabilizers.PhiIdentity()

    model = ks.models.load_model(f"./model_weights/{weights_name}_{suffix}_{kernel_type}.h5", custom_objects={'SSIM': SSIM})
    Psi = reconstructors.StabilizedReconstructor(model, phi)

    x_nn = Psi(y)
    x_corr_nn = Psi(y_delta)

## ----------------------------------------------------------------------------------------------
## ---------- Gauss Filter ----------------------------------------------------------------------
## ----------------------------------------------------------------------------------------------
test_gauss = True
if test_gauss:
    # Reconstruct with StNN
    weights_name = 'gauss_unet'
    sigma = 1
    phi = stabilizers.GaussianFilter(sigma)

    model = ks.models.load_model(f"./model_weights/{weights_name}_{suffix}_{kernel_type}.h5", custom_objects={'SSIM': SSIM})
    Psi = reconstructors.StabilizedReconstructor(model, phi)

    x_gauss = Psi(y)
    x_corr_gauss = Psi(y_delta)

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

    model = ks.models.load_model(f"./model_weights/{weights_name}_{suffix}_{kernel_type}.h5", custom_objects={'SSIM': SSIM})
    Psi = reconstructors.StabilizedReconstructor(model, phi)

    x_tik = Psi(y)
    x_corr_tik = Psi(y_delta)

# Quanitative results
if test_nn and test_gauss and test_stnn:
    # Compute the accuracy
    acc_nn = np.linalg.norm(x_true.flatten() - x_nn.flatten())
    acc_gauss = np.linalg.norm(x_true.flatten() - x_gauss.flatten())
    acc_tik = np.linalg.norm(x_true.flatten() - x_tik.flatten())

    # Compute the stability
    e = y_delta - y
    e_norm = np.linalg.norm(e.flatten())
    
    corr_err_nn = np.linalg.norm(x_true.flatten() - x_corr_nn.flatten())
    corr_err_gauss = np.linalg.norm(x_true.flatten() - x_corr_gauss.flatten())
    corr_err_tik = np.linalg.norm(x_true.flatten() - x_corr_tik.flatten())

    # Print ressults
    import tabulate
    data = [["", "Acc", "Corr_Err", "C"],
            ["NN", acc_nn, corr_err_nn, (corr_err_nn - acc_nn) / e_norm],
            ["Gauss", acc_gauss, corr_err_gauss, (corr_err_gauss - acc_gauss) / e_norm],
            ["Tik", acc_tik, corr_err_tik, (corr_err_tik - acc_tik) / e_norm]]

    print(e_norm)
    print(tabulate.tabulate(data))