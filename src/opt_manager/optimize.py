"""High-level optimization orchestration.

Initializes database connectivity, executes the stochastic branch-and-bound
algorithm, and commits experiment metadata and solutions to persistent storage.
"""

import time
from typing import List, Tuple

from db_manager.driver import initialize_db
from opt_manager.stochastic import branch_bound_algorithm, initialize_attributes


def optimize(
    problem: dict,
    method: dict,
    seeds: List[int],
) -> Tuple[int, float, dict]:
    """Commit problem to db and the branch & bound method.

    Parameters
    ----------
    problem : dict
        The problem definition.
    method : dict
        The method parameters.
    seeds : list[int]
        List of seeds for random number generation, by default None.

    Returns
    -------
    experiment_id : int
        The ID of the experiment.
    elapsed_time : float
        The elapsed time for the optimization process.
    solution : dict
        The solution obtained from the branch & bound algorithm.
    """
    # Initialize db and get db_driver methods
    (
        push_experiment_db,
        push_iteration_db,
        update_exp_time_db,
        push_solution_db,
        close_db,
    ) = initialize_db()

    # Initialize attributes & push experiment to db
    attributes = initialize_attributes(problem, method)
    experiment_id = push_experiment_db(seeds, attributes)

    # Run SB&B algorithm
    start_time = time.perf_counter()
    solution = branch_bound_algorithm(attributes, push_iteration_db)
    elapsed_time = time.perf_counter() - start_time
    update_exp_time_db(elapsed_time)

    # Push solution to database
    push_solution_db(solution)

    # Close the database session
    close_db()

    return experiment_id, elapsed_time, solution
