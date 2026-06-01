"""Activity crashing parameter generation.

Generates random crashing times (as percentages of normal duration) and crashing
costs for project activities, with boundary activities (start/end) set to zero.
"""

from collections import deque

import numpy as np


def generate_crash_times(
    no_of_nodes: int,
    low_limit: float,
    high_limit: float,
) -> list[float]:
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
    list[float]
        List of generated crash times.
    """
    crash_time_deque = deque(np.random.uniform(low_limit, high_limit, no_of_nodes))
    crash_time_deque.appendleft(0.0)
    crash_time_deque.append(0.0)

    crash_time = list(crash_time_deque)

    return crash_time


def generate_crash_cost(
    no_of_nodes: int,
    low_cost: float,
    high_cost: float,
) -> list[float]:
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
    list[float]
        List of generated crash costs.
    """
    crash_cost_deque = deque(np.random.uniform(low_cost, high_cost, no_of_nodes))
    crash_cost_deque.appendleft(0.0)
    crash_cost_deque.append(0.0)

    crash_cost = list(crash_cost_deque)

    return crash_cost
