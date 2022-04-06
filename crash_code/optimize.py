"""Commit problem to db and B&B method."""

import time

import stochastic
from crash_code.db_driver import initialize_db


def optimize(problem, method, seeds=None):
    """Commits problem to db and the branch & bound method."""
    # Initialize db and get db_driver methods
    (
        push_experiment,
        push_iteration,
        update_exp_time,
        push_solution_db,
        close_db,
    ) = initialize_db()

    # Initialize attributes & push experimnt to db
    attributes = stochastic.initialize_attributes(problem, method)
    experiment_id = push_experiment(seeds, attributes)

    # Run SB&B algorithm
    start_time = time.clock()
    solution = stochastic.branch_bound_algorithm(attributes, push_iteration)
    elapsed_time = time.clock() - start_time
    update_exp_time(elapsed_time)

    # Push solution to database
    push_solution_db(solution)

    # Close the database session
    close_db()

    return experiment_id, elapsed_time, solution
