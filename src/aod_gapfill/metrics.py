from __future__ import annotations
import numpy as np
from tensorflow.keras import backend as K
from skimage.metrics import structural_similarity as ssim


# Keras-compatible


def RMSE(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


# NumPy helpers


def rmse_map(y_true: np.ndarray, y_pred: np.ndarray):
    return np.sqrt(np.mean((y_pred - y_true) ** 2, axis=(1,2)))


def ssim_map(y_true: np.ndarray, y_pred: np.ndarray):
    out = np.zeros(y_true.shape[0])
    for i in range(y_true.shape[0]):
        out[i] = ssim(y_true[i], y_pred[i], data_range=max(1e-12, y_true[i].max()-y_true[i].min()))
    return out


def denorm(arr: np.ndarray, vmin: float, vmax: float):
    return arr * (vmax - vmin) + vmin