import numpy as np


def information_gain(parent_entropy, child_entropy):
    return parent_entropy - child_entropy


def entropy(y_true):
    hist = np.bincount(y_true)
    ps = hist / len(y_true)
    return -np.sum([p * np.log(p) for p in ps if p > 0])
