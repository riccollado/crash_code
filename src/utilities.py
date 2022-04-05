# Utility functions:
import numpy as np


def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)


def is_pos_semi_def(x):
    return np.all(np.linalg.eigvals(x) >= 0)


def correlation_from_covariance(covariance):
    v = np.sqrt(np.diag(covariance))
    outer_v = np.outer(v, v)
    correlation = covariance / outer_v
    correlation[covariance == 0] = 0
    return correlation, v


def covariance_from_correlation(correlation, v):
    outer_v = np.outer(v, v)
    covariance = correlation * outer_v
    return covariance
