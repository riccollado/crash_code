"""Generate crashing times and costs."""

from collections import deque

from numpy.random import uniform


def generate_crash_times(no_of_nodes, low_limit, high_limit):
    """Generates activity crash times in percentage.

    Args:
        no_of_nodes (_type_): _description_
        low_limit (_type_): _description_
        high_limit (_type_): _description_

    Returns:
        _type_: _description_
    """
    crash_time = deque(uniform(low_limit, high_limit, no_of_nodes))
    crash_time.appendleft(0.0)
    crash_time.append(0.0)

    crash_time = list(crash_time)

    return crash_time


def generate_crash_cost(no_of_nodes, low_cost, high_cost):
    """Generates activity crash costs.

    Args:
        no_of_nodes (_type_): _description_
        low_cost (_type_): _description_
        high_cost (_type_): _description_

    Returns:
        _type_: _description_
    """
    crash_cost = deque(uniform(low_cost, high_cost, no_of_nodes))
    crash_cost.appendleft(0.0)
    crash_cost.append(0.0)

    crash_cost = list(crash_cost)

    return crash_cost
