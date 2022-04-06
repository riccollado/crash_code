"""Utility functions."""

import numpy as np


def is_pos_def(x):
    """Evaluate positive-definiteness."""
    return np.all(np.linalg.eigvals(x) > 0)


def is_pos_semi_def(x):
    """Evaluate positive-semi-definiteness."""
    return np.all(np.linalg.eigvals(x) >= 0)


def correlation_from_covariance(covariance):
    """Get correlation from covariance."""
    v = np.sqrt(np.diag(covariance))
    outer_v = np.outer(v, v)
    correlation = covariance / outer_v
    correlation[covariance == 0] = 0
    return correlation, v


def covariance_from_correlation(correlation, v):
    """Get covariance from correlation."""
    outer_v = np.outer(v, v)
    covariance = correlation * outer_v
    return covariance
