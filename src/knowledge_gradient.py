from functools import partial

import numpy as np
from scipy.stats import norm


def f_func(z):
    """f function required for KG evaluation

    Parameters
    ----------
    z : float

    Returns
    ----------
    float : float
    """
    return norm.pdf(z) + z * norm.cdf(z)


def sigma(S, x, lambda_):
    """Sigma function

    Parameters
    ----------
    S : np.array
       Covariance matrix
    x : int
       Alternative coordinate
    lambda_ : numpy.ndarray
       Alternative's known variances

    Returns
    ----------
    numpy.ndarray : numpy.ndarray
       Vector calculation of \Sigma function
    """
    # Here we have to remember that x indexing goes from 1 to M
    M = S.shape[0]
    e_x = np.zeros([M, 1])
    e_x[x - 1] = 1

    return np.dot(S, e_x) / np.sqrt(lambda_[x - 1] + S[x - 1][x - 1])


def update_mu_S(mu_n, S_n, lambda_, x, y_n1):
    """Get updated mu and S after taking a sample y at subproblem with name x

    Parameters
    ----------
    mu_n :
    S :
    lambda_ :
    x :
    y :

    Returns
    ----------
    mu_1 :
    S_1 :
    """
    # Get e_x: Adjusted indexing starts from 0 here
    M = S_n.shape[0]
    e_x = np.zeros([M, 1])
    e_x[x] = 1

    mu_n1 = mu_n.reshape(M, 1) + (
        (y_n1 - mu_n[x]) / (lambda_[x] + S_n[x][x]) * np.dot(S_n, e_x)
    )

    # S_n1 = S_n - (1/(lambda_[x]*S_n[x][x]))*S_n.dot(e_x).dot(e_x.transpose()).dot(S_n)
    S_n = S_n - (1 / (lambda_[x] + S_n[x][x])) * S_n.dot(e_x).dot(e_x.transpose()).dot(
        S_n
    )

    return mu_n1, S_n


def algorithm_1(a, b):
    """Algorithm 1 from paper

    Parameters
    ----------
    a : list
       Vector of numbers
    b : list
       Strictly increasing vector of numbers

    Returns
    ----------
    c : list
       Vector output of algorim 1
    A : list
       Set output of Algorithm 1
    """
    M = len(a)
    c = []
    A = []

    # Initialization
    c.append(-np.inf)
    c.append(np.inf)
    A.append(1)
    loop_done = False

    # Main loop
    for i in range(1, M):
        c.append(np.inf)

        while loop_done == False:
            j = A[len(A) - 1]

            c[j] = (a[j - 1] - a[i + 1 - 1]) / (b[i + 1 - 1] - b[j - 1])

            if len(A) != 1 and c[j] <= c[A[len(A) - 1 - 1]]:
                A.pop()
                loop_done = False
            else:
                loop_done = True
        A.append(i + 1)
        loop_done = False

    return c, A


def KG_Alg(mu, S, lambda_):
    """KG algorithm: single-threaded

    Parameters

    ----------
    mu :
    S :
    lambda :

    Returns
    ----------
    xx :
    vv :
    """
    M = S.shape[0]
    xx = -1
    vv = -np.inf

    for x in range(1, M):
        a = list(mu)
        b = list(sigma(S, x, lambda_)[:, 0])

        # Lexicographically sort a and b (with b first)
        data = list(zip(a, b))
        data.sort(key=lambda pair: (pair[1], pair[0]))
        a, b = (list(t) for t in zip(*data))

        for i in range(1, M - 1):
            if b[i - 1] == b[i + 1 - 1]:
                # Remove entry i-1 from data pairs sequences
                del data[i - 1]
                del b[i - 1]
                del a[i - 1]
        # Use Algorithm 1 to obtain c and A
        c, A = algorithm_1(a, b)

        # Obtain restricions of a, b, c to
        # indexes from A (we have to adjust
        # due to the index disparity of c, a
        # starting from 1 but a, b start from 0)
        # From now on a, b, c, A have the same index set!!!
        a = [a[j - 1] for j in A]
        b = [b[j - 1] for j in A]
        c = [c[j] for j in A]
        M = len(A)

        v = np.log(
            np.sum([(b[j + 1] - b[j]) * f_func(-np.abs(c[j])) for j in range(0, M - 1)])
        )

        if x == 1 or v > vv:
            vv = v
            xx = x

    return xx - 1, vv


def KG_iteration(mu, S, lambda_, x):
    """Single KG iteration

    Parameters
    ----------
    mu :
    S :
    lambda :

    Returns
    ----------
    xx :
    vv :
    """
    M = S.shape[0]
    a = list(mu)
    b = list(sigma(S, x, lambda_)[:, 0])

    # Lexicographically sort a and b (with b first)
    data = list(zip(a, b))
    data.sort(key=lambda pair: (pair[1], pair[0]))
    a, b = (list(t) for t in zip(*data))

    for i in range(1, M - 1):
        if b[i - 1] == b[i + 1 - 1]:
            # Remove entry i-1 from data
            # pairs sequences
            del data[i - 1]
            del b[i - 1]
            del a[i - 1]
    # Use Algorithm 1 to obtain c and A
    c, A = algorithm_1(a, b)

    # Obtain restricions of a, b, c to
    # indexes from A (we have to adjust
    # due to the index disparity of c, a
    # starting from 1 but a, b start from 0)
    # From now on a, b, c, A have the same index set!!!
    a = [a[j - 1] for j in A]
    b = [b[j - 1] for j in A]
    c = [c[j] for j in A]
    M = len(A)

    v = np.log(
        np.sum([(b[j + 1] - b[j]) * f_func(-np.abs(c[j])) for j in range(0, M - 1)])
    )

    return x - 1, v


def KG_multi(mu, S, lambda_, pool):
    """KG algorithm: parallelilized for large instances

    Parameters
    ----------
    mu :
    S :
    lambda :
    pool :

    Returns
    ----------
    L : list
    """
    M = S.shape[0]

    if M >= 10:
        L = pool.map(
            partial(
                KG_iteration,
                mu,
                S,
                lambda_,
            ),
            range(1, M),
        )

    else:
        L = []
        for x in range(1, M):
            L.append(KG_iteration(mu, S, lambda_, x))

    L.sort(key=lambda pair: pair[1], reverse=True)

    return L[0]
