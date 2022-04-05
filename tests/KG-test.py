"""A simple test of KG method."""

import multiprocessing as mp

import numpy as np
from scipy.stats import random_correlation

from knowledge_gradient import KG_Alg, KG_multi, update_mu_S
from nearest_correlation import nearcorr

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
        n_pos_eig=0,
        weights=None,
        verbose=False,
        except_on_too_many_iterations=True,
    )

    M = S.shape[0]

    lambda_ = np.array([0.2, 1.1, 1.3, 0.12, 0.4, 0.3, 0.12])

    mu = np.array([0.2, 0.21, 0.92, 0.11, 0.7, 0.2, -0.1])

    print(KG_Alg(mu, S, lambda_))

    print(KG_multi(mu, S, lambda_, pool))

    y = 0.22
    x = 3
    mu_1, S_1 = update_mu_S(mu, S, lambda_, x, y)
    print(mu_1.shape)
    print(mu_1)

    print(S_1.shape)
    print(S_1)
