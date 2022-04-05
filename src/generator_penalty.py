from numpy import power

from uncrashed_bounds import uncrashed_project_time


def generate_penalty_vals_linear(t, m, b1):
    """Generates penalty linear values
    Parameters
    ----------
    t : list
       Points at which we evaluate the penalty step function
    m : float
       Line slope value
    b1 : float
       Line y intercept

    Returns
    ----------
    vals : list
       Liear values of the form m*t[i] + b1 (except val[0] which is always zero)
    """
    vals = [0]
    vals.extend([m * t[i] + b1 for i in range(1, len(t))])
    return vals


def generate_penalty_vals_exponential(t, m, b1):
    """Generates penalty exponential values
    Parameters
    ----------
    t : list
       Points at which we evaluate the penalty step function
    m : float
       multiplier
    b1 : float
       base

    Returns
    ----------
    vals : list
       Exponential values of the form m*b1^(t[i]) (except val[0] which is always zero)
    """
    vals = [0]
    vals.extend([m * power(b1, t[i]) for i in range(1, len(t))])
    return vals


def generate_penalty_bounds(network, PERT):
    # Obtain t_init and t_final for calculation of objective
    # penalty function. This is done by solving the main problem
    # without crashing with most likely and pesimistic scenarios.
    # In this way we approximate 'normal-time' and 'worst-time'
    # it takes to perform the project. We use this to establish
    # boundaries for the penalty function.
    t_init, _ = uncrashed_project_time(network, PERT["most_likely"])
    t_final, _ = uncrashed_project_time(network, PERT["pessimistic"])

    return [t_init, t_final]
