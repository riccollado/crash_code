"""
A simple test of the Knowledge Gradient (KG) method.

This script tests the functionality of the KG_Alg, KG_multi, and update_mu_S functions
from the opt_manager.knowledge_gradient module. It generates a random correlation
matrix, adjusts it to be the nearest correlation matrix, and then applies the KG methods
to update the mean vector and covariance matrix based on the given parameters.

Functions tested:
- KG_Alg: Computes the Knowledge Gradient for a single decision.
- KG_multi: Computes the Knowledge Gradient for multiple decisions using
multiprocessing.
- update_mu_S: Updates the mean vector and covariance matrix after observing a new data
point.
"""

import multiprocessing as mp

import numpy as np
from scipy.stats import random_correlation

from matrix_manager.nearest_correlation import nearcorr
from opt_manager.knowledge_gradient import kg_alg, kg_multi, update_mu_s

if __name__ == "__main__":
    processes = mp.cpu_count()
    pool = mp.Pool(processes)

    np.random.seed(126)

    g = 7 / sum([0.5, 0.8, 1.2, 2.5, 1.7, 2.1, 2.2])
    G = np.round(
        random_correlation.rvs(
            (g * 0.5, g * 0.8, g * 1.2, g * 2.5, g * 1.7, g * 2.1, g * 2.2)
        ),
        3,
    )

    S = nearcorr(
        G,
        tol=[],
        flag=0,
        max_iterations=1000,
        weights=None,
        except_on_too_many_iterations=True,
    )

    M = S.shape[0]

    lambda_values = np.array([0.2, 1.1, 1.3, 0.12, 0.4, 0.3, 0.12])

    mu = np.array([0.2, 0.21, 0.92, 0.11, 0.7, 0.2, -0.1])

    print(kg_alg(mu, S, lambda_values))

    print(kg_multi(mu, S, lambda_values, pool))

    Y = 0.22
    X = 3
    updated_mu, updated_S = update_mu_s(mu, S, lambda_values, X, Y)
    print(updated_mu.shape)
    print(updated_mu)

    print(updated_S.shape)
    print(updated_S)

    pool.close()
    pool.join()
