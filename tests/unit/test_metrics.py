import pytest
import numpy as np
from tinyML import commons


@pytest.mark.parametrize(
    "y_true, y_pred, exp_mse",
    [
        (np.array([1, 2, 3]), np.array([1, 2, 3]), 0),
        (np.array([1, 2, 3]), np.array([4, 5, 6]), 9),
        (np.array([-1, -2, -3]), np.array([-1, -2, -3]), 0),
        (np.array([1.5, 2.5, 3.5]), np.array([1, 2, 3]), 0.25),
        (np.array([1e10, 2e10, 3e10]), np.array([1.1e10, 2.1e10, 3.1e10]), 1e18),
    ],
)
def test_mse(y_true, y_pred, exp_mse):
    assert commons.mse(y_true, y_pred) == exp_mse


@pytest.mark.parametrize(
    "y_true, y_pred",
    [
        (np.array([1, 2, 3]), np.array([4, 5])),
        (np.array([]), np.array([])),
    ],
)
def test_mse_exceptions(y_true, y_pred):
    with pytest.raises(ValueError):
        commons.mse(y_true, y_pred)
