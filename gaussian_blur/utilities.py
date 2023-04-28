import tensorflow as tf
from tensorflow import keras as ks

from IPPy.metrics import *
from IPPy.nn import models

import numpy as np
import matplotlib.pyplot as plt

def psnr_loss(y_true, y_pred):
    """
    PSNR Loss. As in the paper: Simple Baseline for Image Restoration.
    """
    return tf.reduce_mean(tf.math.log(tf.reduce_mean((y_true - y_pred) ** 2, axis=(1, 2, 3))) + 1e-8)

def load_model_by_parameters(recon_name, model_type, suffix, kernel_type):
    """
    Given the parameters for the model, returns the corresponding model.s
    """
    # Define the weights full path
    weights_name = f"./gaussian_blur/model_weights/{recon_name}_{model_type}_{suffix}_{kernel_type}.h5"
    return ks.models.load_model(weights_name, custom_objects={'SSIM': SSIM, 'psnr_loss': psnr_loss})

def build_model(model_type):
    if model_type == 'unet':
        model = models.get_UNet(input_shape = (256, 256, 1), n_scales = 4, conv_per_scale = 2, final_relu=True, skip_connection=False)
    elif model_type == 'ssnet':
        model = models.get_SSNet(input_shape = (256, 256, 1), n_ch=(128, 128, 128), k_size=(9, 5, 3), final_relu=True, skip_connection=False)
    elif model_type == 'baseline':
        model = models.get_BaselineModel(input_shape = (256, 256, 1), conv_per_scale=[3, 3, 3], init_conv=32)
    elif model_type == 'nafnet':
        model = models.get_NAFModel(input_shape = (256, 256, 1), conv_per_scale=[3, 3, 3], init_conv=32)
    return model

def RGB2Gray(x, inverse=False):
    """
    If x = h x w x c is an RGB image, this function returns x as an array c x h x w x 1.
    If x = c x h x w x 1 is a grey-scale image, this function returns x as an array h x w x c (if the flag "inverse" is True).
    """
    if inverse:
        return np.transpose(x[:, :, :, 0], (1, 2, 0))
    return np.expand_dims(np.transpose(x, (2, 0, 1)), -1)

def normalize(x):
    """
    Normalize x in [0, 1].
    """
    return (x - x.min() ) / (x.max() - x.min())

def generate_error_plot(error_path, delta_min=0, delta_max=1, suffix=''):
    errors = np.load(error_path)
    xaxis = np.linspace(delta_min, delta_max, len(errors)) #* 256

    plt.figure()
    plt.plot(xaxis, errors[0],'o-')
    plt.plot(xaxis, errors[1],'o-')
    plt.plot(xaxis, errors[2],'o-')
    plt.grid()
    plt.legend(['NN', 'FiNN', 'StNN'])
    plt.xlabel(r'$\delta$')
    plt.ylabel(r'$\|\| \Psi(Kx + e) - x \|\|$')
    plt.axis(ymin=0.05, ymax=0.30)
    plt.tight_layout()
    plt.title('Test case A.1')
    plt.xticks(xaxis, rotation=30)
    plt.savefig(f'relerrors_ssnet_{suffix}_{delta_max}_EE.png', dpi=300)

def corr_error_over_noise_amplitude(model, model_type, noise_level, delta):
    """
    The idea is to make a (scatter) plot with ||e|| in abscissae, || Psi(Ax+e) - x || - eta on y-axis. 
    As a result, the points below the y=x line are "stable", while points above the y=x line are unstable.
    """
    # Compute the maximum x-lim
    Mx = (delta + noise_level) * 256

    # Load data in memory
    suffix = str(noise_level).split('.')[-1]
    delta_suffix = str(delta).split('.')[-1]

    clean_err = np.load(f"./gaussian_blur/results/clean_err_{model}_{model_type}_{suffix}_{delta_suffix}_gaussian.npy")
    corr_err = np.load(f"./gaussian_blur/results/corr_err_{model}_{model_type}_{suffix}_{delta_suffix}_gaussian.npy")
    norm_e = np.load(f"./gaussian_blur/results/norm_e_{model}_{model_type}_{suffix}_{delta_suffix}_gaussian.npy")

    # Compute the accuracy
    inv_acc = np.max(clean_err)
    
    # Get the color
    c = ['r' if (corr_err[i]- inv_acc)>norm_e[i] else 'green' for i in range(len(corr_err))]

    plt.figure()
    plt.scatter(norm_e, corr_err - inv_acc, s=5, c=c)
    plt.plot((0, Mx), (0, Mx), 'b--')

    plt.xlim((0, Mx))
    # plt.ylim((0, 3*Mx))
    plt.xlabel(r"$|| e_i ||$")
    plt.ylabel(r"$|| \Psi(Ax_i + e_i) - x_i || - \eta$")

    plt.grid()
    plt.tight_layout()
    plt.savefig(f"./gaussian_blur/results/corr_error_over_noise_amplitude_{model}_{model_type}_{suffix}_{delta_suffix}_gaussian.png", dpi=300)
    plt.close()

def plot_local_stability_constant_test_set(model, model_type, noise_level, delta):
    """
    The idea is to make a (scatter) plot with i = 1, ..., N in abscissae, the local stability constant on y-axis.
    """
    # Load data in memory
    suffix = str(noise_level).split('.')[-1]
    delta_suffix = str(delta).split('.')[-1]

    clean_err = np.load(f"./gaussian_blur/results/clean_err_{model}_{model_type}_{suffix}_{delta_suffix}_gaussian.npy")
    corr_err = np.load(f"./gaussian_blur/results/corr_err_{model}_{model_type}_{suffix}_{delta_suffix}_gaussian.npy")
    norm_e = np.load(f"./gaussian_blur/results/norm_e_{model}_{model_type}_{suffix}_{delta_suffix}_gaussian.npy")

    # Compute the accuracy
    inv_acc = np.max(clean_err)

    # Compute the local stability constant
    C = (corr_err - inv_acc) / norm_e
    
    # Get the color
    c = ['r' if C[i]>1 else 'green' for i in range(len(corr_err))]

    plt.figure()
    plt.scatter(np.arange(len(corr_err)), C, s=5, c=c)
    plt.plot((0, len(corr_err)-1), (1, 1), 'b--')

    plt.ylim((-3, 3))
    plt.xlabel("i")
    plt.ylabel(r"$C^\delta_\Psi(i)$")

    plt.grid()
    plt.tight_layout()
    plt.savefig(f"./gaussian_blur/results/local_stability_constant_{model}_{model_type}_{suffix}_{delta_suffix}_gaussian.png", dpi=300)
    plt.close()

def visualize_boxplot(model_type, noise_level, delta, loss):
    """
    Generate boxplots as in the experiment.
    """
    # Load data in memory
    suffix = str(noise_level).split('.')[-1]
    delta_suffix = '_g_noise'+str(delta).split('.')[-1] if delta != 0 else ''
     
    loss_vec = np.load(f"./gaussian_blur/results/{loss}_{model_type}_{suffix}{delta_suffix}_gaussian.npy")

    # Remove y_delta
    loss_vec = loss_vec[:, 1:]

    # Reflect
    loss_vec = loss_vec[:, ::-1]

    # Draw boxplot
    ax = plt.axes()
    bplot = plt.boxplot(loss_vec, vert=False, patch_artist=True, notch=True)
    plt.yticks([1, 2, 3], ['StNN', 'FiNN', 'NN'])
    
    plt.xlim((0.5, 1))
    plt.xlabel(loss)

    # fill with colors
    colors = ['lightblue', 'orange', 'lightgreen'][::-1]
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)

    # adding horizontal grid lines
    ax.xaxis.grid(True)

    plt.tight_layout()
    plt.savefig(f"./gaussian_blur/results/boxplot_{loss}_{model_type}_{suffix}{delta_suffix}_gaussian.png", dpi=400)
    plt.close()


corr_error_over_noise_amplitude('nn', 'nafnet', 0, 0.01)
corr_error_over_noise_amplitude('finn', 'nafnet', 0, 0.01)
corr_error_over_noise_amplitude('stnn', 'nafnet', 0, 0.01)

plot_local_stability_constant_test_set('nn', 'nafnet', 0, 0.01)
plot_local_stability_constant_test_set('finn', 'nafnet', 0, 0.01)
plot_local_stability_constant_test_set('stnn', 'nafnet', 0, 0.01)

visualize_boxplot('nafnet', 0, 0, 'SSIM')
visualize_boxplot('nafnet', 0, 0.01, 'SSIM')

corr_error_over_noise_amplitude('nn', 'nafnet', 0.025, 0.05)
corr_error_over_noise_amplitude('finn', 'nafnet', 0.025, 0.05)
corr_error_over_noise_amplitude('stnn', 'nafnet', 0.025, 0.05)

plot_local_stability_constant_test_set('nn', 'nafnet', 0.025, 0.05)
plot_local_stability_constant_test_set('finn', 'nafnet', 0.025, 0.05)
plot_local_stability_constant_test_set('stnn', 'nafnet', 0.025, 0.05)

visualize_boxplot('nafnet', 0.025, 0, 'SSIM')
visualize_boxplot('nafnet', 0.025, 0.05, 'SSIM')