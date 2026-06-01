"""End-to-end single experiment runner.

CLI entry point that generates a random project-crashing problem instance,
configures the stochastic branch-and-bound solver with specified method
parameters, executes the optimization, and displays the solution.
"""

import multiprocessing as mp
import os
import platform
import random
from typing import Any

import numpy as np
from pyfiglet import Figlet

from gen_manager.covariance import generate_cov_mat
from gen_manager.crash import generate_crash_cost, generate_crash_times
from gen_manager.distribution import generate_geometric, generate_pert_distributions
from gen_manager.network import generate_network
from gen_manager.penalty import generate_penalty_bounds
from opt_manager.optimize import optimize

if __name__ == "__main__":
    processes = mp.cpu_count()
    pool = mp.Pool(int(processes * 3 / 4))

    # Set seeds
    SEED_1 = 281323
    SEED_2 = 945672
    seeds = [SEED_1, SEED_2]
    random.seed(SEED_1)
    np.random.seed(SEED_2)

    # ----------------------------------------------------------
    # Randomly generate a problem
    # ----------------------------------------------------------
    problem = {}
    NO_OF_NODES = 25
    NO_OF_LAYERS = 5
    DENSITY = 0.4
    (
        problem["network"],
        problem["network_figure"],
        problem["network_pos"],
    ) = generate_network(NO_OF_NODES, NO_OF_LAYERS, DENSITY)
    problem["cov_mat"] = generate_cov_mat(NO_OF_NODES)

    # Generate geometric probabilities
    geom_prob = generate_geometric(NO_OF_NODES)

    # Generate PERT optimistic, mostlikely, and pessimistic
    # activity durations
    problem["PERT"] = generate_pert_distributions(geom_prob)

    # Generate crash times & costs
    LOW_LIMIT = 0.1
    HIGH_LIMIT = 0.5
    LOW_COST = 100
    HIGH_COST = 200
    problem["crash_time"] = generate_crash_times(NO_OF_NODES, LOW_LIMIT, HIGH_LIMIT)
    problem["crash_cost"] = generate_crash_cost(NO_OF_NODES, LOW_COST, HIGH_COST)

    # Generate penalty function
    penalty: dict[str, int | float | str | Any] = {}
    penalty["type"] = "linear"  # or "exponential"
    penalty["steps"] = 20.0
    penalty["m"] = 15.0
    penalty["b1"] = 21.0
    penalty["t_init"], penalty["t_final"] = generate_penalty_bounds(
        problem["network"], problem["PERT"]
    )
    problem["penalty"] = penalty

    # ----------------------------------------------------------
    # Set method parameters
    # ----------------------------------------------------------
    method: dict[str, int | float | bool | str | mp.pool.Pool] = {}
    method["pool"] = pool
    method["type"] = "KG"

    # Branch & Bound parameters
    method["scenarios_per_estimation"] = 10
    method["total_scenarios"] = 1_000

    # Bootstrap parameters
    method["bootstrap"] = True
    method["resamples"] = 50
    method["confidence"] = 0.05

    # Pareto method parameters
    method["pareto_beta"] = 1.5

    # KG method parameters
    method["KG_sigma"] = 0.5
    method["KG_l"] = 3.0

    # ----------------------------------------------------------
    # Solve problem
    # ----------------------------------------------------------
    experiment_id, elapsed_time, solution = optimize(problem, method, seeds)

    # ----------------------------------------------------------
    # Output solution to screen
    # ----------------------------------------------------------
    if platform.system() == "Windows":
        os.system("cls")
    else:
        os.system("clear")

    print(Figlet(font="big").renderText("Crashing Cost"))

    print(f"Experiment ID: {experiment_id}")
    print(f"Elapsed time: {elapsed_time:.2f} sec")
    print(f"Expected cost: {solution['E_solution']:.2f}")
    print(f"Standard deviation: {solution['Std_sol']:.2f}")
    print("\n\n")
