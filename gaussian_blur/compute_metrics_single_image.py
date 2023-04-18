import numpy as np
import matplotlib.pyplot as plt
from skimage import data

from PIL import Image

from tensorflow import keras as ks

from IPPy.nn import models as NN_models
from IPPy.nn.datasets import *
from IPPy import reconstructors
from IPPy.metrics import *
from IPPy.utils import *
from IPPy.operators import *
from IPPy import stabilizers

import utilities

import argparse
import yaml
parser = argparse.ArgumentParser()
parser.add_argument("-p", "--path",
                    help="Path to the image you want to process. If an int is given, then the corresponding test image will be processed.",
                    required=True)
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
# Load the test image
if args.path.isdigit():
    idx = int(args.path)
    x_true = test_data[idx]
else:
    # Load the given image
    x_true = Image.open(args.path)[:, :, 0]
    x_true = np.array(x_true.resize((256, 256)))

K = ConvolutionOperator(kernel, (m, n))
y = K @ x_true
y_delta = y.reshape((m, n)) + (noise_level + delta) * np.random.normal(0, 1, (m, n))

# Poisson
if peak != 0:
    y_delta = np.random.poisson(y_delta * peak) / peak

# Save the results
plt.imsave(f'./gaussian_blur/results/corr_{suffix}_{out_domain_label}{kernel_type}.png', y_delta.reshape((m, n)), cmap='gray', dpi=400)

re_vec = [rel_err(y_delta, x_true)]
psnr_vec = [PSNR(y_delta, x_true)]
ssim_vec = [SSIM(y_delta, x_true).numpy()]
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

    x_rec = Psi(y_delta)

    # Quantitative results
    re_vec.append(rel_err(x_rec, x_true))
    psnr_vec.append(PSNR(x_true, x_rec))
    ssim_vec.append(SSIM(x_true, x_rec).numpy())

    # Save the results
    plt.imsave(f'./gaussian_blur/results/{weights_name}_{suffix}_{out_domain_label}{kernel_type}_recon.png', x_rec.reshape((m, n)), cmap='gray', dpi=400)


# Print out quantitative results
import tabulate
data = [["", "Start", ] + args.model,
        ["RE"] + re_vec,
        ["PSNR"] + psnr_vec,
        ["SSIM"] + ssim_vec]

print(tabulate.tabulate(data))