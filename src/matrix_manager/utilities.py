"""Matrix definiteness checks and covariance-correlation conversion.

Provides utilities for testing positive-(semi)definiteness of matrices and
converting between covariance and correlation representations.
"""

import numpy as np
from numpy.typing import NDArray

def is_pos_def(x: NDArray[np.float64]) -> bool:
    """Evaluate positive-definiteness.

    Parameters
    ----------
    x : numpy.ndarray
        The matrix to evaluate.

    Returns
    -------
    bool
        True if the matrix is positive-definite, False otherwise.
    """
    return np.all(np.linalg.eigvals(x) > 0)

def is_pos_semi_def(x: NDArray[np.float64]) -> bool:
    """Evaluate positive-semi-definiteness.

    Parameters
    ----------
    x : numpy.ndarray
        The matrix to evaluate.

    Returns
    -------
    bool
        True if the matrix is positive-semi-definite, False otherwise.
    """
    return np.all(np.linalg.eigvals(x) >= 0)

def correlation_from_covariance(
    covariance: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Get correlation from covariance.

    Parameters
    ----------
    covariance : numpy.ndarray
        The covariance matrix.

    Returns
    -------
    correlation : numpy.ndarray
        The correlation matrix.
    v : numpy.ndarray
        The standard deviations of the variables.
    """
    v = np.sqrt(np.diag(covariance))
    outer_v = np.outer(v, v)
    correlation = covariance / outer_v
    correlation[covariance == 0] = 0
    return correlation, v

def covariance_from_correlation(
    correlation: NDArray[np.float64], v: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Get covariance from correlation.

    Parameters
    ----------
    correlation : numpy.ndarray
        The correlation matrix.
    v : numpy.ndarray
        The standard deviations of the variables.

    Returns
    -------
    covariance : numpy.ndarray
        The covariance matrix.
    """
    outer_v = np.outer(v, v)
    covariance = correlation * outer_v
    return covariance
