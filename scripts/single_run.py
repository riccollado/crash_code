"""Execute a single run of main method."""

import multiprocessing as mp
import os
import platform
import random

import numpy as np
from pyfiglet import Figlet

from crash_code.generator_covariance import generate_cov_mat
from crash_code.generator_crash import generate_crash_cost, generate_crash_times
from crash_code.generator_distribution import generate_geometric, generate_PERT
from crash_code.generator_network import generate_network
from crash_code.generator_penalty import generate_penalty_bounds
from crash_code.optimize import optimize

if __name__ == "__main__":

    processes = mp.cpu_count()
    pool = mp.Pool(int(processes * 3 / 4))

    # Set seeds
    seed1 = 281323
    seed2 = 945672
    seeds = [seed1, seed2]
    random.seed(seed1)
    np.random.seed(seed2)

    # ----------------------------------------------------------
    # Randomly generate a problem
    # ----------------------------------------------------------
    problem = {}
    no_of_nodes = 25
    no_of_layers = 5
    density = 0.4
    (
        problem["network"],
        problem["network_figure"],
        problem["network_pos"],
    ) = generate_network(no_of_nodes, no_of_layers, density)
    problem["cov_mat"] = generate_cov_mat(no_of_nodes)

    # Generate geometric probabilities
    geom_prob = generate_geometric(no_of_nodes)

    # Generate PERT optimistic, mostlikely, and pessimistic
    # activity durations
    problem["PERT"] = generate_PERT(geom_prob)

    # Generate crash times & costs
    low_limit = 0.1
    high_limit = 0.5
    low_cost = 100
    high_cost = 200
    problem["crash_time"] = generate_crash_times(no_of_nodes, low_limit, high_limit)
    problem["crash_cost"] = generate_crash_cost(no_of_nodes, low_cost, high_cost)

    # Generate penalty function
    penalty = {}
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
    method = {}
    method["pool"] = pool
    method["type"] = "KG"

    # Branch & Bound parameters
    method["scenarios_per_estimation"] = 10
    method["total_scenarios"] = 1000

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

    print("Experiment ID: {}".format(experiment_id))
    print("Elapsed time: {:.2f} sec".format(elapsed_time))
    print("Expected cost: {:.2f}".format(solution["E_solution"]))
    print("Standard deviation: {:.2f}".format(solution["Std_sol"]))
    print("\n\n")
