import numpy as np


def mse(y_true, y_pred):
    if len(y_true) != len(y_pred):
        raise ValueError("The inputs are not of the same length.")
    if len(y_true) * len(y_pred) == 0:
        raise ValueError("One or both of the inputs are of length 0.")

    return np.mean(np.square(y_true - y_pred))


def binary_cross_entropy(y_true, y_pred):
    epsilon = 1e-15  # to avoid log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def accuracy(y_true, y_pred):
    return np.mean(y_pred == y_true)
