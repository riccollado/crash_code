from collections import deque

from numpy.random import uniform


def generate_crash_times(no_of_nodes, low_limit, high_limit):
    """Generates activity crash times in percentage

    Parameters
    ----------
    no_of_nodes : int
       Number of nodes (activities) in the network

    Returns
    ----------
    N/A
    """
    crash_time = deque(uniform(low_limit, high_limit, no_of_nodes))
    crash_time.appendleft(0.0)
    crash_time.append(0.0)

    crash_time = list(crash_time)

    return crash_time


def generate_crash_cost(no_of_nodes, low_cost, high_cost):
    """Generates activity crash costs

    Parameters
    ----------
    no_of_nodes : int
       Number of nodes (activities) in the network

    Returns
    ----------
    N/A
    """
    crash_cost = deque(uniform(low_cost, high_cost, no_of_nodes))
    crash_cost.appendleft(0.0)
    crash_cost.append(0.0)

    crash_cost = list(crash_cost)

    return crash_cost
