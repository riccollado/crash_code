"""Random correlation matrix generation.

Constructs symmetric positive-semidefinite correlation matrices by projecting
randomly populated matrices onto the nearest correlation matrix using Higham's
algorithm.
"""

import secrets

import numpy as np

from matrix_manager.nearest_correlation import nearcorr


def generate_cov_mat(size: int) -> np.ndarray:
    """Generate a random size x size  correlation matrix.

    Parameters
    ----------
    size : int
       Dimension of the square correlation matrix

    Returns
    -------
    corr_mat : numpy.ndarray
       Correlation matrix
    """
    rand_mat = np.zeros((size, size))
    for k in range(size):
        for i in range(size):
            if k == i:
                rand_mat[k][i] = 1
            elif k < i:
                rand_mat[k][i] = secrets.SystemRandom().uniform(-1, 1)
            else:
                rand_mat[k][i] = rand_mat[i][k]
    corr_mat = nearcorr(
        symmetric_input_matrix=rand_mat,
        tol=[],
        flag=0,
        max_iterations=50000,
        weights=None,
        except_on_too_many_iterations=True,
    )
    return corr_mat
