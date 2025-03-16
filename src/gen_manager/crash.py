"""Generate crashing times and costs."""

from collections import deque
from typing import List

from numpy.random import uniform


def generate_crash_times(
    no_of_nodes: int,
    low_limit: float,
    high_limit: float,
) -> List[float]:
    """Generate activity crash times in percentage.

    Parameters
    ----------
    no_of_nodes : int
        Number of nodes.
    low_limit : float
        Lower limit for crash times.
    high_limit : float
        Upper limit for crash times.

    Returns
    -------
    List[float]
        List of generated crash times.
    """
    crash_time_deque = deque(uniform(low_limit, high_limit, no_of_nodes))
    crash_time_deque.appendleft(0.0)
    crash_time_deque.append(0.0)

    crash_time = list(crash_time_deque)

    return crash_time


def generate_crash_cost(
    no_of_nodes: int,
    low_cost: float,
    high_cost: float,
) -> List[float]:
    """Generate activity crash costs.

    Parameters
    ----------
    no_of_nodes : int
        Number of nodes.
    low_cost : float
        Lower limit for crash costs.
    high_cost : float
        Upper limit for crash costs.

    Returns
    -------
    List[float]
        List of generated crash costs.
    """
    crash_cost_deque = deque(uniform(low_cost, high_cost, no_of_nodes))
    crash_cost_deque.appendleft(0.0)
    crash_cost_deque.append(0.0)

    crash_cost = list(crash_cost_deque)

    return crash_cost
