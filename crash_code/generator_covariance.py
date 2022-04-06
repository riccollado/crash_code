"""Generate covariance matrix."""

import random

import numpy as np

from crash_code.nearest_correlation import nearcorr


def generate_cov_mat(size):
    """Generates a random size x size  correlation matrix.

    Parameters
    ----------
    size : int
       Dimension of the square correlation matrix

    Returns
    ----------
    A : numpy.ndarray
       Correlation matrix
    """
    P = np.zeros((size, size))
    for k in range(size):
        for i in range(size):
            if k == i:
                P[k][i] = 1
            elif k < i:
                P[k][i] = random.uniform(-1, 1)
            else:
                P[k][i] = P[i][k]
    A = nearcorr(
        P,
        tol=[],
        flag=0,
        max_iterations=50000,
        n_pos_eig=0,
        weights=None,
        verbose=False,
        except_on_too_many_iterations=True,
    )
    return A
