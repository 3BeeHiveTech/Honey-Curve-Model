"""
Interpolation functions for numpy arrays.
"""

import numpy as np


def linear_interp_1d(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Linear interpolation of two 1-d numpy arrays.

    Args:
        x: 1-d numpy array of x values.
        y: 1-d numpy array of y values.

    Returns:
        beta_hat: (2,) array containing the y-axis intercept and slope of the linear regression
            model.
        yinterp: 1-d numpy array of the interpolated y values.
    """

    assert x.ndim == 1, "x must be a 1-d numpy array."
    assert y.ndim == 1, "y must be a 1-d numpy array."

    # Calculate interpolation with matrix formula
    # bhat = (x^T * x )^(-1) *  x^T * y
    x_mat = np.vstack((np.ones(len(x)), x)).T
    beta_hat = np.linalg.inv(x_mat.T.dot(x_mat)).dot(x_mat.T).dot(y)

    # Calculate the interpolated y values with dot product
    yinterp = x_mat.dot(beta_hat)

    return beta_hat, yinterp
