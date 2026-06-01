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
from typing import Any

class ExceededMaxIterationsError(Exception):
    """Error class for exceeding iterations."""

    def __init__(
        self,
        msg: str,
        matrix: np.ndarray | list[Any] | None = None,
        iteration: int | list[int] | None = None,
        ds: np.ndarray | list[Any] | None = None,
    ) -> None:
        """Initialize restart state for near-correlation iterations.

        Parameters
        ----------
        msg : str
            Error message describing why the iteration stopped.
        matrix : numpy.ndarray or list, optional
            Last candidate matrix produced by the algorithm.
        iteration : int or list of int, optional
            Iteration count reached before stopping.
        ds : numpy.ndarray or list, optional
            Last correction matrix used by the alternating projection.

        Returns
        -------
        None
        """
        if matrix is None:
            matrix = []
        if iteration is None:
            iteration = []
        if ds is None:
            ds = []
        self.msg = msg
        self.matrix = matrix
        self.iteration = iteration
        self.ds = ds

    def __str__(self) -> str:
        """Provide a printable representation of the error.

        Returns
        -------
        str
            Stored error message.
        """
        return repr(self.msg)

def nearcorr(
    symmetric_input_matrix: np.ndarray | ExceededMaxIterationsError,
    tol: list[float] | np.ndarray | None = None,
    flag: int = 0,
    max_iterations: int = 100,
    weights: list[float] | np.ndarray | None = None,
    except_on_too_many_iterations: bool = True,
) -> np.ndarray:
    """Compute the nearest correlation matrix for a symmetric input matrix.

    Parameters
    ----------
    symmetric_input_matrix : numpy.ndarray or ExceededMaxIterationsError
        Symmetric matrix to be projected, or an error object containing a restart
        state from a previous run.
    tol : list of float or numpy.ndarray, optional
        Convergence tolerance. If omitted, the implementation default is used.
    flag : int, default=0
        Projection mode. ``0`` uses full eigendecomposition. ``1`` is reserved and
        currently not implemented.
    max_iterations : int, default=100
        Maximum number of alternating-projection iterations.
    weights : list of float or numpy.ndarray, optional
        Diagonal weights for the weighted Frobenius norm projection.
    except_on_too_many_iterations : bool, default=True
        Whether to raise an exception when convergence is not achieved within
        ``max_iterations``.

    Returns
    -------
    numpy.ndarray
        Projected nearest correlation matrix.

    Raises
    ------
    ValueError
        If the input matrix is not symmetric.
    ExceededMaxIterationsError
        If convergence is not achieved and
        ``except_on_too_many_iterations`` is True.
    NotImplementedError
        If ``flag`` is set to 1.
    """
    if tol is None:
        tol = []
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

def proj_spd(A: np.ndarray) -> np.ndarray:
    """Project a symmetric matrix onto the positive semidefinite cone.

    Parameters
    ----------
    A : numpy.ndarray
        Symmetric matrix to project.

    Returns
    -------
    numpy.ndarray
        Positive semidefinite matrix obtained after clipping negative eigenvalues.
    """
    # NOTE: the input matrix is assumed to be symmetric
    d, v = np.linalg.eigh(A)
    A = (v * np.maximum(d, 0)).dot(v.T)
    A = (A + A.T) / 2
    return A
