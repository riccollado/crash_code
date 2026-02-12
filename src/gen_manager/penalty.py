"""Penalty function generation and bounds computation.

Provides linear and exponential penalty functions for project deadline overruns,
and computes penalty bounds by solving uncrashed baseline scenarios.
"""

from typing import Any, Dict, List

from numpy import power

from opt_manager.uncrashed_bounds import uncrashed_project_time


def generate_penalty_vals_linear(
    t: List[float],
    m: float,
    b1: float,
) -> List[float]:
    """Generate penalty linear values.

    Parameters
    ----------
    t : list of float
        Points at which we evaluate the penalty step function.
    m : float
        Line slope value.
    b1 : float
        Line y intercept.

    Returns
    -------
    vals : list of float
        Linear values of the form m*t[i] + b1 (except val[0] which is always zero).
    """
    vals: List[float] = [0]
    vals.extend([m * t[i] + b1 for i in range(1, len(t))])
    return vals


def generate_penalty_vals_exponential(
    t: List[float],
    m: float,
    b1: float,
) -> List[float]:
    """Generate penalty exponential values.

    Parameters
    ----------
    t : list of float
        Points at which we evaluate the penalty step function.
    m : float
        Multiplier.
    b1 : float
        Base.

    Returns
    -------
    vals : list of float
        Exponential values of the form m * b1^(t[i]) (except val[0] which is always
        zero).
    """
    vals: List[float] = [0]
    vals.extend([m * power(b1, t[i]) for i in range(1, len(t))])
    return vals


def generate_penalty_bounds(
    network: Any,
    pert_dist: Dict[str, List[float]],
) -> List[float]:
    """Obtain t_init and t_final for calculation of objective penalty function.

    This is done by solving the main problem without crashing with most likely
    and pessimistic scenarios. In this way we approximate 'normal-time' and
    'worst-time' it takes to perform the project. We use this to establish
    boundaries for the penalty function.

    Parameters
    ----------
    network : Any
        The network representing the project.
    pert_dist : dict of str to list of float
        Dictionary containing 'most_likely' and 'pessimistic' scenarios.

    Returns
    -------
    list of float
        A list containing t_init and t_final values.
    """
    t_init, _ = uncrashed_project_time(network, pert_dist["most_likely"])
    t_final, _ = uncrashed_project_time(network, pert_dist["pessimistic"])
    return [t_init, t_final]
