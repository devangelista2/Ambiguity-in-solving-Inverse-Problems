import tensorflow as tf
from tensorflow import keras as ks

from IPPy.metrics import *

def psnr_loss(y_true, y_pred):
    return tf.reduce_mean(tf.math.log(tf.reduce_mean((y_true - y_pred) ** 2, axis=(1, 2, 3))) + 1e-8)

def load_model_by_parameters(recon_name, model_type, suffix, kernel_type):
    # Define the weights full path
    weights_name = f"./model_weights/{recon_name}_{model_type}_{suffix}_{kernel_type}.h5"
    return ks.models.load_model(weights_name, custom_objects={'SSIM': SSIM, 'psnr_loss': psnr_loss})