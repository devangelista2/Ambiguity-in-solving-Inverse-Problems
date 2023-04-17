import numpy as np
import matplotlib.pyplot as plt
from skimage import data

from tensorflow import keras as ks

import IPPy.nn.models as NN_models
from IPPy import reconstructors
from IPPy.metrics import *
from IPPy.utils import *
from IPPy.operators import *
from IPPy.nn.datasets import *
from IPPy import stabilizers


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

noise_level = 0.0
delta = 0.030 # Out-of-domain noise intensity

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
x_true = train_data[700]
K = ConvolutionOperator(kernel, (m, n))
y = K @ x_true
y = y.reshape((m, n))

errors = np.zeros((3, 11))
for i, d in enumerate(np.linspace(0, delta, 11)):
    y_delta = y + d * np.random.normal(0, 1, y.shape)

    ## ----------------------------------------------------------------------------------------------
    ## ---------- NN --------------------------------------------------------------------------------
    ## ----------------------------------------------------------------------------------------------
    weights_name = 'nn_ssnet'
    phi = stabilizers.PhiIdentity()

    model = ks.models.load_model(f"./model_weights/{weights_name}_{suffix}_{kernel_type}.h5", custom_objects={'SSIM': SSIM})
    Psi = reconstructors.StabilizedReconstructor(model, phi)

    x_nn = Psi(y_delta)

    ## ----------------------------------------------------------------------------------------------
    ## ---------- Gauss Filter ----------------------------------------------------------------------
    ## ----------------------------------------------------------------------------------------------
    # Reconstruct with StNN
    weights_name = 'gauss_ssnet'
    sigma = 1
    phi = stabilizers.GaussianFilter(sigma)

    model = ks.models.load_model(f"./model_weights/{weights_name}_{suffix}_{kernel_type}.h5", custom_objects={'SSIM': SSIM})
    Psi = reconstructors.StabilizedReconstructor(model, phi)

    x_gauss = Psi(y_delta)

    ## ----------------------------------------------------------------------------------------------
    ## ---------- StNN ------------------------------------------------------------------------------
    ## ----------------------------------------------------------------------------------------------
    # Reconstruct with StNN
    weights_name = 'tik_ssnet'
    if noise_level == 0:
        reg_param = 1e-2
    elif noise_level == 0.025:
        reg_param = 1e-2
    phi = stabilizers.Tik_CGLS_stabilizer(kernel, reg_param, k=3)

    model = ks.models.load_model(f"./model_weights/{weights_name}_{suffix}_{kernel_type}.h5", custom_objects={'SSIM': SSIM})
    Psi = reconstructors.StabilizedReconstructor(model, phi)

    x_tik = Psi(y_delta)

    # Append errors
    errors[0, i] = np.linalg.norm(x_nn.flatten() - x_true.flatten())/np.linalg.norm(x_true.flatten())
    errors[1, i] = np.linalg.norm(x_gauss.flatten() - x_true.flatten())/np.linalg.norm(x_true.flatten())
    errors[2, i] = np.linalg.norm(x_tik.flatten() - x_true.flatten())/np.linalg.norm(x_true.flatten())
    print(f"Iteration {i+1}.")
np.save(f'relerrors_ssnet_{suffix}_{delta}.npy', errors)