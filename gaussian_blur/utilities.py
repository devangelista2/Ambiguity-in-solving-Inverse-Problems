import tensorflow as tf
from tensorflow import keras as ks

from IPPy.metrics import *
from IPPy.nn import models

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