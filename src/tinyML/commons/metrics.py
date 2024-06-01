import numpy as np


def mse(y_true, y_pred):
    if len(y_true) != len(y_pred):
        raise ValueError("The inputs are not of the same length.")
    if len(y_true) * len(y_pred) == 0:
        raise ValueError("One or both of the inputs are of length 0.")

    return np.mean(np.square(y_true - y_pred))
