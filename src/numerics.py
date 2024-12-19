"""Everything related to arrays/numbers."""

import numpy as np


def one_hot_encode(values, n_classes: int):
    n = len(values)
    k = n_classes
    m = np.zeros((n, k))
    row_indices = np.arange(n)
    m[row_indices, values] = 1.0
    return m


def normalize_values(x, vmin: float, vmax: float):
    return np.clip((x - vmin) / (vmax - vmin), 0, 1.0)


def cast_as_2d(x):
    x = np.array(x)
    if x.ndim == 0:
        return np.array([[x]])
    elif x.ndim == 1:
        return x.reshape(-1, 1)
    return x
