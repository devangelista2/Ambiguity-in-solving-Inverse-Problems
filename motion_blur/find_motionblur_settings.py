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
DATA_PATH = './data/'
TRAIN_PATH = os.path.join(DATA_PATH, 'GOPRO_train_small.npy')

train_data = np.load(TRAIN_PATH)
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
y_delta = y.reshape((m, n)) + noise_level * np.random.normal(0, 1, (m, n)) + delta * np.random.normal(0, 1, (m, n)) 

# Save the results
plt.imsave(f'./motionblur_tests/corr_{suffix}_{out_domain_label}{kernel_type}.png', y_delta.reshape((m, n)), cmap='gray', dpi=400)

# Reconstruct by skimage filters
from skimage import filters
from skimage import restoration

x_gauss = filters.gaussian(y_delta)
x_tv = restoration.denoise_tv_chambolle(y_delta, 0.3, max_num_iter = 10)

# Save the results
plt.imsave(f'./motionblur_tests/gauss_filter_{suffix}_{out_domain_label}{kernel_type}_recon.png', x_gauss.reshape((m, n)), cmap='gray', dpi=400)
plt.imsave(f'./motionblur_tests/tv_filter_{suffix}_{out_domain_label}{kernel_type}_recon.png', x_tv.reshape((m, n)), cmap='gray', dpi=400)

## Reconstruct with algorithm
#weights_name = 'is'
#reg_param = 1
#phi = stabilizers.Tik_CGLS_stabilizer(kernel, reg_param, k=10)
#
#x_is = phi(y_delta)
#print(reg_param, SSIM(x_true, x_is))
#
## Save the results
#plt.imsave(f'./motionblur_tests/{weights_name}_{suffix}_{out_domain_label}{kernel_type}_recon.png', x_is.reshape((m, n)), cmap='gray', dpi=400)