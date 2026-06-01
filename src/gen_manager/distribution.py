"""PERT activity-time distribution generation.

Constructs Beta distributions for project activity durations using the PERT
(Program Evaluation and Review Technique) three-point estimation method with
optimistic, most-likely, and pessimistic times sampled via geometric distributions.
"""

import secrets
from typing import Any

import numpy as np
from scipy.stats import beta


def generate_pert_distributions(geom_prob: dict[Any, float]) -> dict[str, Any]:
    """Generate beta distributions for activity times.

    Parameters
    ----------
    geom_prob : dict
        Dictionary with nodes as keys (int) and geometric probabilities as values
        (float). The probabilities are used to sample the optimistic, most-likely,
        and pessimistic activity-time parameters.

    Returns
    -------
    dict[str, Any]
        Dictionary with:

        - ``"distributions"``: mapping from node id to frozen
          ``scipy.stats.beta`` distributions.
        - ``"optimistic"``: list of optimistic activity durations.
        - ``"most_likely"``: list of most-likely activity durations.
        - ``"pessimistic"``: list of pessimistic activity durations.
    """
    # Set maximum iterations for sub-loops
    max_iterations = 10_000

    distributions = {}

    optimistic_duration = []
    most_likely_duration = []
    pessimistic_duration = []

    # Activity duration of start node is zero
    optimistic_duration.append(0)
    most_likely_duration.append(0)
    pessimistic_duration.append(0)

    for key, probability in geom_prob.items():
        # Choosing optimistic, optimistic_mlikely_diff, pessimistic_mlikely_diff from
        # geometric distributions
        o_value = 0
        ml_value = 0
        p_value = 0

        iterations = 0
        while (o_value == 0 or o_value >= 20) and iterations < max_iterations:
            iterations += 1
            o_value = np.random.geometric(probability) + 5

        optimistic_duration.append(o_value)
        o_ml_diff = np.random.geometric(probability)
        p_ml_diff = np.random.geometric(probability)

        # Obtain most likely and pessimistic time for activities based on above sampled
        # values. Restrict the most likely value to be below 20 and pessimistic below
        # 100
        # i = random.randint(1, 5)
        i = secrets.randbelow(5) + 1

        iterations = 0
        while (ml_value <= 0 or ml_value > 20) and iterations < max_iterations:
            iterations += 1
            ml_value = o_value + i * o_ml_diff
            o_ml_diff = np.random.geometric(probability)
            # i = random.randint(1, 5)
            i = secrets.randbelow(5) + 1
        most_likely_duration.append(ml_value)

        iterations = 0
        while (p_value <= 0 or p_value > 100) and iterations < max_iterations:
            iterations += 1
            p_value = ml_value + (i * 5) * p_ml_diff
            p_ml_diff = np.random.geometric(probability)
            # i = random.randint(1, 5)
            i = secrets.randbelow(5) + 1
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


def generate_geometric(no_of_nodes: int) -> dict[int, float]:
    """Generate probabilities for geometric distributions.

    Used to select the projects betas.

    Parameters
    ----------
    no_of_nodes : int
        Number of nodes in the network graph.

    Returns
    -------
    dict of int to float
        Dictionary with node ids as keys and random geometric probabilities as
        values.
    """
    # geom_prob = {i: random.uniform(0.001, 1) for i in range(1, no_of_nodes + 1)}
    secure_random = secrets.SystemRandom()
    geom_prob = {i: secure_random.uniform(0.001, 1) for i in range(1, no_of_nodes + 1)}
    return geom_prob
