"""Generate PERT and geometric distributions."""

import random

import numpy as np
from scipy.stats import beta


def generate_PERT(geom_prob):
    """Generate beta distributions for activity times.

    Parameters
    ----------
    geom_prob : dictionary
       Probabilities for Geometric distributions used to
       select the activities' Beta distributions

    Returns
    ----------
    distributions : dictionary
       Dictionary with nodes as keys and values the corresponding
       beta distribution of activity times.
    """
    distributions = {}

    optimistic_duration = []
    most_likely_duration = []
    pessimistic_duration = []

    # Activity duration of start node is zero
    optimistic_duration.append(0)
    most_likely_duration.append(0)
    pessimistic_duration.append(0)

    for key, probability in geom_prob.items():
        # Choosing optimistic, optimistic_mlikely_diff,
        # pessimestic_mlikely_diff from geometric distributions
        o_value = 0
        ml_value = 0
        p_value = 0

        while o_value == 0 or o_value >= 20:
            o_value = np.random.geometric(probability) + 5

        optimistic_duration.append(o_value)
        o_ml_diff = np.random.geometric(probability)
        p_ml_diff = np.random.geometric(probability)

        # Obtain most likely and pessimistic time for
        # activities based on above sampled values.
        # Restrict the most likely value to be below
        # 20 and pessimistic below 100
        i = random.randint(1, 5)
        while ml_value <= 0 or ml_value > 20:
            ml_value = o_value + i * o_ml_diff
            o_ml_diff = np.random.geometric(probability)
            i = random.randint(1, 5)
        most_likely_duration.append(ml_value)

        while p_value <= 0 or p_value > 100:
            p_value = ml_value + (i * 5) * p_ml_diff
            p_ml_diff = np.random.geometric(probability)
            i = random.randint(1, 5)
        pessimistic_duration.append(p_value)

        a = o_value
        m = ml_value
        b = p_value

        # Calculating alpha and beta from PERT
        alpha = 1 + 4 * (m - a) / (b - a)
        bet = 1 + 4 * (b - m) / (b - a)

        # Calculate pert beta distribution using alpha and bet
        distributions[key] = beta(alpha, bet, loc=a, scale=b - a)

    # Activity duration of end node is zero
    optimistic_duration.append(0)
    most_likely_duration.append(0)
    pessimistic_duration.append(0)

    return {
        "distributions": distributions,
        "optimistic": optimistic_duration,
        "most_likely": most_likely_duration,
        "pessimistic": pessimistic_duration,
    }


def generate_geometric(no_of_nodes):
    """Generates probabilities for geometric distributions.

    Used to select the projects betas.

    Parameters
    ----------
    no_of_nodes : int
       Number of nodes in the network graph

    Returns
    ----------
    geom_prob : dictionary
       Dictionary with nodes as keys and values the corresponding
       geometric distribution probability.
    """
    geom_prob = {}
    for i in range(1, no_of_nodes + 1):
        value = random.uniform(0, 1)
        geom_prob[i] = value

    return geom_prob
