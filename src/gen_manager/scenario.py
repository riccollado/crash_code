"""Correlated scenario generation via Gaussian copula.

Generates samples of correlated activity durations using a Gaussian copula with
specified correlation structure and marginal PERT distributions.
"""

from typing import Any

import numpy as np
from scipy.stats import norm


def dynamic_scenarios(
    num_activities: int,
    num_scenarios: int,
    correlation_matrix: np.ndarray,
    distribution_dict: dict[int, Any],
) -> np.ndarray:
    """Generate samples based on activities and correlation matrix.

    Parameters
    ----------
    num_activities : int
        Number of activities.
    num_scenarios : int
        Number of scenarios to generate.
    correlation_matrix : numpy.ndarray
        Correlation matrix.
    distribution_dict : dict
        Dictionary with nodes as keys and values the corresponding
        beta distribution of activity times.

    Returns
    -------
    sample_array : numpy.ndarray
        Numpy array with samples (columns=activities, rows=samples).
    """
    # Generate samples via Gaussian Copula
    mean = np.zeros(num_activities)
    rng = np.random.default_rng()
    samples = norm.cdf(
        rng.multivariate_normal(
            mean=mean,
            cov=correlation_matrix,
            size=num_scenarios,
        )
    )
    k_list = []

    for i in range(num_activities):
        k = np.array(distribution_dict[i + 1].ppf(samples[:, i]))
        k_list.append(k)

    # Start node duration = 0
    sample_array = np.zeros(num_scenarios)

    # End node duration = 0
    k_list.append(np.zeros(num_scenarios))

    for k in k_list:
        sample_array = np.column_stack((sample_array, k))

    return sample_array
