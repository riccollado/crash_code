"""Find nearest correlation matrix.

This file and its methods are obtained from Nick Higham's implementation of the nearest
correlation matrix algorithm. The original implementation can be found at:
https://www.maths.manchester.ac.uk/~higham/narep/nearcorr.m and
https://nhigham.com/2013/02/13/the-nearest-correlation-matrix/ .

For this reason, this file will be excluded from test coverage and from PEP8, MyPy, and
PyLint checks.
"""

import numpy as np
from numpy import copy, inf
from numpy.linalg import norm


class ExceededMaxIterationsError(Exception):
    """Error class for exceeding iterations."""

    def __init__(self, msg, matrix=[], iteration=[], ds=[]):
        """Initialize instance."""
        self.msg = msg
        self.matrix = matrix
        self.iteration = iteration
        self.ds = ds

    def __str__(self):
        """Provide description string for instance."""
        return repr(self.msg)


def nearcorr(
    symmetric_input_matrix,
    tol=[],
    flag=0,
    max_iterations=100,
    weights=None,
    except_on_too_many_iterations=True,
):
    """Find nearest correlation.

    X = nearcorr(symmetric_input_matrix, tol=[], flag=0, max_iterations=100,
    weights=None, print=0)

    Finds the nearest correlation matrix to the symmetric matrix symmetric_input_matrix.

    symmetric_input_matrix is a symmetric numpy array or a ExceededMaxIterationsError
    object.

    tol is a convergence tolerance, which defaults to 16*EPS. If using flag == 1, tol
    must be a size 2 tuple, with first component the convergence tolerance and second
    component a tolerance for defining "sufficiently positive" eigenvalues.

    flag = 0: solve using full eigen decomposition (EIG). flag = 1: treat as "highly
    non-positive definite A" and solve using partial eigen decomposition (EIGS).
    CURRENTLY NOT IMPLEMENTED

    max_iterations is the maximum number of iterations (default 100, but may need to be
    increased).

    weights is an optional vector defining a diagonal weight matrix diag(W).

    except_on_too_many_iterations = True to raise an exception when number of iterations
    exceeds max_iterations except_on_too_many_iterations = False to silently return the
    best result found after max_iterations number of iterations
    """
    # If input is an ExceededMaxIterationsError object this
    # is a restart computation
    if isinstance(symmetric_input_matrix, ExceededMaxIterationsError):
        ds = copy(symmetric_input_matrix.ds)
        symmetric_input_matrix = copy(symmetric_input_matrix.matrix)
    else:
        ds = np.zeros(np.shape(symmetric_input_matrix))

    eps = np.spacing(1)
    if not np.all((np.transpose(symmetric_input_matrix) == symmetric_input_matrix)):
        raise ValueError("Input Matrix is not symmetric")
    if not tol:
        tol = eps * np.shape(symmetric_input_matrix)[0] * np.array([1, 1])
    if weights is None:
        weights = np.ones(np.shape(symmetric_input_matrix)[0])
    X = copy(symmetric_input_matrix)
    Y = copy(symmetric_input_matrix)
    rel_diffY = inf
    rel_diffX = inf
    rel_diffXY = inf

    Whalf = np.sqrt(np.outer(weights, weights))

    iteration = 0
    while max(rel_diffX, rel_diffY, rel_diffXY) > tol[0]:
        iteration += 1
        if iteration > max_iterations:
            if except_on_too_many_iterations:
                if max_iterations == 1:
                    message = (
                        "No solution found in " + str(max_iterations) + " iteration"
                    )
                else:
                    message = (
                        "No solution found in " + str(max_iterations) + " iterations"
                    )
                raise ExceededMaxIterationsError(message, X, iteration, ds)
            else:
                # exceptOnTooManyIterations is false so just silently
                # return the result even though it has not converged
                return X

        Xold = copy(X)
        R = X - ds
        R_wtd = Whalf * R
        if flag == 0:
            X = proj_spd(R_wtd)
        elif flag == 1:
            raise NotImplementedError(
                "Setting 'flag' to 1 is currently\
                                 not implemented."
            )
        X = X / Whalf
        ds = X - R
        Yold = copy(Y)
        Y = copy(X)
        np.fill_diagonal(Y, 1)
        normY = norm(Y, "fro")
        rel_diffX = norm(X - Xold, "fro") / norm(X, "fro")
        rel_diffY = norm(Y - Yold, "fro") / normY
        rel_diffXY = norm(Y - X, "fro") / normY

        X = copy(Y)

    return X


def proj_spd(A):
    """Projected SPD."""
    # NOTE: the input matrix is assumed to be symmetric
    d, v = np.linalg.eigh(A)
    A = (v * np.maximum(d, 0)).dot(v.T)
    A = (A + A.T) / 2
    return A
