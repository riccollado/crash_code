"""Main stochastic optimization module."""

import math
import time
from functools import partial

import bootstrapped.bootstrap as bs
import bootstrapped.stats_functions as bs_stats
import numpy as np
import pandas as pd
import pygmo as pg
from jellyfish import levenshtein_distance as l_dist
from statsmodels.stats.correlation_tools import cov_nearest

from crash_code.generator_scenario import dynamic_scenarios
from crash_code.generator_subproblem import optimize_subproblem
from crash_code.knowledge_gradient import KG_multi, update_mu_S

KG_leaf_counter = 0
COV = pd.DataFrame()


def increment_lc():
    """Increment leaf counter."""
    global KG_leaf_counter
    KG_leaf_counter += 1
    return KG_leaf_counter


def initialize_attributes(problem, method):
    """Initializes common attributes and parameters needed for SB&B algorithm.

    Returns
    ----------
    attributes : dict
       Dictionary with keys:
          pool, project_network, scenarios, crashtime, crashcost, t_init, t_final,
          outlocation, b, b2, alpha, experiment_id, conn, method,
          num_scenerios_per_estimation, beta, pessimistic_duration, penalty_type,
          m, b1
    """
    attributes = {}

    # Network attributes
    attributes["network"] = problem["network"]
    attributes["nodes"] = list(problem["network"].nodes)
    attributes["no_of_nodes"] = problem["network"].number_of_nodes()
    attributes["no_of_edges"] = problem["network"].number_of_edges()

    # Network Figure
    attributes["binary_figure"] = problem["network_figure"]
    attributes["pos_figure"] = problem["network_pos"]

    # Time distribution attributes
    attributes["cov_mat"] = problem["cov_mat"]
    attributes["distributions"] = problem["PERT"]["distributions"]
    attributes["optimistic"] = problem["PERT"]["optimistic"]
    attributes["most_likely"] = problem["PERT"]["most_likely"]
    attributes["pessimistic"] = problem["PERT"]["pessimistic"]

    # Crash times & costs
    attributes["crash_time"] = problem["crash_time"]
    attributes["crash_cost"] = problem["crash_cost"]

    # Penalty attributes
    attributes["penalty_type"] = problem["penalty"]["type"]
    attributes["m"] = problem["penalty"]["m"]
    attributes["b1"] = problem["penalty"]["b1"]
    attributes["penalty_steps"] = problem["penalty"]["steps"]
    attributes["t_init"] = problem["penalty"]["t_init"]
    attributes["t_final"] = problem["penalty"]["t_final"]

    # Method attributes
    attributes["pool"] = method["pool"]
    attributes["method_type"] = method["type"]

    # Branch & Bound attributes
    attributes["scen_est_num"] = method["scenarios_per_estimation"]
    attributes["total_scenarios"] = method["total_scenarios"]

    # Bootstrap attributes
    attributes["bootstrap"] = method["bootstrap"]
    attributes["resamples"] = method["resamples"]
    attributes["confidence"] = method["confidence"]

    # Pareto method attributes
    attributes["pareto_beta"] = method["pareto_beta"]

    # KG method attributes
    attributes["KG_sigma"] = method["KG_sigma"]
    attributes["KG_l"] = method["KG_l"]
    attributes["KG_mu"] = {}
    attributes["KG_lambda"] = {}

    # Scenario placeholder
    attributes["scenarios"] = []

    # Some global definitions
    global KG_leaf_counter
    KG_leaf_counter = 0
    global COV
    COV = pd.DataFrame()

    return attributes


def branch_bound_algorithm(attributes, push_iteration):
    """Stochastic Branch & Bound implementation."""
    global COV

    # Step 0: Scenario space initialization
    # We need some extra generated scenarios but
    # we do not want to ocunt them in the total ussage
    scen_est_num = attributes["scen_est_num"]
    total_scenarios = attributes["total_scenarios"] + 4 * scen_est_num
    attributes["scenarios"] = dynamic_scenarios(
        attributes["no_of_nodes"],
        total_scenarios,
        attributes["cov_mat"],
        attributes["distributions"],
    )
    total_scenarios -= 4 * scen_est_num

    # Step 1: Initialization
    subproblem = {}
    variable_list = []

    for node in attributes["nodes"]:
        variable_list.append("x" + str(node))

    # Create 1st subproblem
    subproblem["variables"] = variable_list
    subproblem["constraints"] = {}
    subproblem["recordset"] = True
    subproblem["sample_solutions"] = []

    if len(variable_list) > 1:
        subproblem["singleton"] = False
    else:
        subproblem["singleton"] = True

    subproblem["E"] = np.inf
    subproblem["STD"] = np.inf
    subproblem["num_sample_solutions"] = 0
    subproblem["total_scenarios_done"] = 0
    subproblem["scenarios_done"] = 0
    subproblem["Z_E"] = np.inf
    subproblem["Z_std"] = np.inf
    subproblem["KG_E"] = np.inf
    subproblem["KG_std"] = np.inf
    subproblem["KG_sample_sol"] = []
    subproblem["name"] = increment_lc()

    # Update COV matrix
    name = subproblem["name"]
    COV[name] = ""
    COV = COV.append([""], ignore_index=True)
    COV.drop([0], axis=1, inplace=True)
    COV.index = np.arange(1, len(COV) + 1)

    # Initial KG beliefs (KG_mu, KG_lambda):
    # We start from single problem so we basically
    # do not have beliefs. Later we would start
    # from a deep tree and there the beliefs would
    # be used. For the moment we'll have to do
    # with this.
    attributes["KG_mu"] = {name: 0}
    attributes["KG_lambda"] = {name: 1}

    # Step 1.2: Partition list of B&B tree
    # leaf sub-problems initialization
    partition_list = []
    partition_list.append(subproblem)

    # Flag to mark when we have just
    # partitioned a record set
    partitioned_flag = False

    # Iteration counter
    iteration = 1

    # Set start/end index keeping track of
    # how many samples are taken (need to be taken)
    scenario_start_index = 0
    scenario_end_index = scenario_start_index + scen_est_num

    # Main recursive loop: Iterate over the
    # total number of scenarions.
    while scenario_end_index <= total_scenarios:
        start_iter_time = time.clock()

        # Step 2: Record set partitioning
        partitioned_flag = partition_record_set(
            attributes, partition_list, scenario_start_index, scenario_end_index
        )

        # Update scenario end-points if we spent extra scenarios on partitioning
        if partitioned_flag:
            scenario_start_index = scenario_start_index + 2 * scen_est_num
            scenario_end_index = scenario_start_index + scen_est_num

        # Step 3: Bound estimation
        work_on_subproblems(
            attributes, partition_list, scenario_start_index, scenario_end_index
        )

        # Update scenario end-points for next iteration
        scenario_start_index = scenario_end_index
        scenario_end_index = scenario_start_index + scen_est_num

        # Push iteration results in DB
        elapsed_iter_time = time.clock() - start_iter_time
        push_iteration(COV, partition_list, attributes, iteration, elapsed_iter_time)

        # Increment iteration and loop
        iteration += 1

    # Get optimal solution (minimal E) among all partitions
    # and get the corresponding variance. If we didn't do
    # bootstrap, the variace values are not so useful.
    return_data = {}
    E_data = [partition_list[i]["E"] for i in range(len(partition_list))]
    return_data["E_solution"] = np.min(E_data)
    return_data["E_data"] = E_data
    return_data["Partial_sol"] = partition_list[E_data.index(min(E_data))][
        "constraints"
    ]
    Std_data = [partition_list[i]["Z_std"] for i in range(len(partition_list))]
    return_data["Std_data"] = Std_data
    return_data["Std_sol"] = Std_data[E_data.index(return_data["E_solution"])]

    return return_data


def partition_record_set(
    attributes, partition_list, scenario_start_index, scenario_end_index
):
    """Partition (split) record set and update bounds on new subproblems.

    Return
    ----------
     just_partitioned : bool
        States if we have just partitoned (split) the record set in two
    """
    global COV

    just_partitioned = False

    for subproblem in partition_list:

        # Partition and update bounds if we have non-singleton record set
        if subproblem["recordset"] is True and subproblem["singleton"] is False:

            # Get constraints List for subproblem
            constraints_dict = subproblem["constraints"]
            index_of_last_constraint = len(constraints_dict) - 1

            # Create 1st split of record set
            subproblem1 = {}
            subproblem1["variables"] = subproblem["variables"].copy()
            subproblem1["constraints"] = subproblem["constraints"].copy()
            subproblem1["constraints"][index_of_last_constraint + 1] = 0
            subproblem1["recordset"] = False
            subproblem1["sample_solutions"] = []
            subproblem1["singleton"] = False
            subproblem1["E"] = np.inf
            subproblem1["STD"] = np.inf
            subproblem1["num_sample_solutions"] = 0
            subproblem1["total_scenarios_done"] = 0
            subproblem1["scenarios_done"] = 0
            subproblem1["Z_E"] = np.inf
            subproblem1["Z_std"] = np.inf
            subproblem1["KG_E"] = np.inf
            subproblem1["KG_std"] = np.inf
            subproblem1["KG_sample_sol"] = []
            subproblem1["name"] = increment_lc()

            # Create 2nd split of record set
            subproblem2 = {}
            subproblem2["variables"] = subproblem["variables"].copy()
            subproblem2["constraints"] = subproblem["constraints"].copy()
            subproblem2["constraints"][index_of_last_constraint + 1] = 1
            subproblem2["recordset"] = False
            subproblem2["sample_solutions"] = []
            subproblem2["singleton"] = False
            subproblem2["E"] = np.inf
            subproblem2["STD"] = np.inf
            subproblem2["num_sample_solutions"] = 0
            subproblem2["total_scenarios_done"] = 0
            subproblem2["scenarios_done"] = 0
            subproblem2["Z_E"] = np.inf
            subproblem2["Z_std"] = np.inf
            subproblem2["KG_E"] = np.inf
            subproblem2["KG_std"] = np.inf
            subproblem2["KG_sample_sol"] = []
            subproblem2["name"] = increment_lc()

            # Mark new singletons
            if index_of_last_constraint == attributes["no_of_nodes"] - 2:
                subproblem1["singleton"] = True
                subproblem2["singleton"] = True

            # Update partition list (list of SB&B tree leafs)
            partition_list.remove(subproblem)
            partition_list.extend([subproblem1, subproblem2])

            # Update COV matrix and beliefs for
            # extra subproblems in KG
            if attributes["method_type"] == "KG":
                KG_mu = attributes["KG_mu"]
                KG_lambda = attributes["KG_lambda"]
                sub_name = subproblem["name"]
                sub1_name = subproblem1["name"]
                sub2_name = subproblem2["name"]

                # Update subproblem beliefs:
                # here we just don't know and set
                # things very simple: mean=0, var=1
                del KG_mu[sub_name]
                KG_mu[sub1_name] = 0.0
                KG_mu[sub2_name] = 0.0
                del KG_lambda[sub_name]
                KG_lambda[sub1_name] = 1.0
                KG_lambda[sub2_name] = 1.0

                # Delete subproblem row and column
                COV.drop([sub_name], inplace=True)
                COV.drop([sub_name], axis=1, inplace=True)

                # Add empty columns for 2 new subproblems
                COV[sub1_name] = ""
                COV[sub2_name] = ""

                # Add two empty rows for new subproblems
                s1 = pd.Series(name=sub1_name)
                s2 = pd.Series(name=sub2_name)
                COV = COV.append(s1)
                COV = COV.append(s2)

                # Update with exponential kernel values
                # for the two new rows and columns and find
                # nearest covariance matrix
                update_row_col(subproblem1, subproblem2, partition_list, attributes)

                # **********COV UPDATING FINISHED**********

            # Estimate bounds by taking full sample scenarios
            # at each new leaf. This step also takes care of
            # priming the KG sample statistics.
            scen_est_num = attributes["scen_est_num"]
            estimate_bounds(
                attributes,
                subproblem1,
                attributes["scenarios"][
                    scenario_start_index : scenario_start_index + scen_est_num
                ],
            )
            estimate_bounds(
                attributes,
                subproblem2,
                attributes["scenarios"][
                    scenario_start_index
                    + scen_est_num : scenario_start_index
                    + 2 * scen_est_num
                ],
            )

            just_partitioned = True
            return just_partitioned

        # Do not split if we have a singleton record set
        elif subproblem["recordset"] is True and subproblem["singleton"] is True:
            just_partitioned = False
            return just_partitioned

    return just_partitioned


def work_on_subproblems(
    attributes, partition_list, scenario_start_index, scenario_end_index
):
    """Obtain assigned scenarios per subproblem and calls bounding method."""
    total_scenarios = attributes["scenarios"][scenario_start_index:scenario_end_index]
    assigned_scenarios = []

    # Assign to each subproblem the scenarios used to
    # update bounds.
    if (
        attributes["method_type"] == "Random"
        or attributes["method_type"] == "Random_1"
        or attributes["method_type"] == "Distance"
    ):
        assigned_scenarios = assign_scenarios(
            partition_list, total_scenarios, attributes["method_type"]
        )

    if (
        attributes["method_type"] == "Pareto_Inverse"
        or attributes["method_type"] == "Pareto_Boltzman"
    ):
        assigned_scenarios = assign_scenarios_pareto(
            partition_list,
            total_scenarios,
            attributes["pareto_beta"],
            attributes["method_type"],
        )

    if attributes["method_type"] == "KG":
        assigned_scenarios = assign_scenarios_KG(
            partition_list, total_scenarios, attributes
        )

    # List to hold newly obtained subproblem results
    min_obj_val = []

    for i, subproblem in enumerate(partition_list):
        subproblem["recordset"] = False

        # No need to assign scenarios in case of Uniform since
        # every subproblem will get assigned ALL scenarios.
        if attributes["method_type"] == "Uniform":
            scenarios = total_scenarios

        else:
            scenarios = assigned_scenarios[i]

        if len(scenarios) != 0:
            # Update bounds by solving optimization problems for each scenario
            min_obj_val.append(estimate_bounds(attributes, subproblem, scenarios))

        else:
            # If no scenarios get allocated then copy the
            # data from last iteration
            min_obj_val.append(subproblem["E"])

    # Setting the record set
    index = min_obj_val.index(min(min_obj_val))
    partition_list[index]["recordset"] = True

    return


def estimate_bounds(attributes, subproblem, scenarios):
    """Call solver on a given subproblem for all scenarios.

    Calculates E, Z_E and Z_std Parameters. Also makes the individual subproblem
    E and STD updates under the KG method.

    Returns
    ----------
    E_obj_value : float
       Approximation to expectation obtained from averaging the solution of
        subproblem over every scenario either applying Boostrap or not.
    """
    global COV

    scenario_solutions = multisolve_scenarios(attributes, subproblem, scenarios)

    # Obtain all scenario solutions and add to subproblem
    sample_solutions = []
    for i, scenario in enumerate(scenarios):
        sample_solutions.append(float(scenario_solutions[i]["Objective func Value"]))
    subproblem["sample_solutions"] += sample_solutions

    # Compute bootstrap statistics if required
    if attributes["bootstrap"] is True:
        subproblem["Z_E"], subproblem["Z_std"] = bootstrap_E_STD(subproblem, attributes)
        # Update mean and std from boostrap
        subproblem["E"] = E_obj_value = subproblem["Z_E"]
        subproblem["STD"] = std_obj_function_value = subproblem["Z_std"]

    else:
        # Calculate mean over all scenarios
        E_obj_value = np.mean(subproblem["sample_solutions"])
        subproblem["E"] = subproblem["Z_E"] = E_obj_value
        # Calculate STD over all scenarios
        std_obj_function_value = np.std(subproblem["sample_solutions"])
        subproblem["STD"] = subproblem["Z_std"] = std_obj_function_value

    subproblem["num_sample_solutions"] = len(subproblem["sample_solutions"])
    subproblem["total_scenarios_done"] = subproblem.get(
        "total_scenarios_done", 0
    ) + len(scenarios)
    subproblem["scenarios_done"] = len(scenarios)

    #  Update the KG sample & Belief statistics
    if attributes["method_type"] == "KG":
        # First the sample statistics:
        subproblem["KG_sample_sol"].append(np.mean(sample_solutions))
        # Apply KG-bootstrap
        if attributes["bootstrap"] is True:
            subproblem["KG_E"], subproblem["KG_std"] = bootstrap_KG(
                subproblem, attributes, sample_solutions
            )
        else:
            subproblem["KG_E"] = np.mean(sample_solutions)
            subproblem["KG_std"] = np.std(sample_solutions)

        # Now update the belief statistics:
        # First setup the vectors and matrices we need
        mu_n = np.fromiter(attributes["KG_mu"].values(), dtype=float)
        S_n = COV.to_numpy(dtype=float, copy=False)
        lambda_ = np.fromiter(attributes["KG_lambda"].values(), dtype=float)
        y_n1 = subproblem["KG_E"]

        idx = list(attributes["KG_lambda"].keys())
        x = idx.index(subproblem["name"])

        # Now, do the operation and make sure COV and
        # KG_lambda and KG_mu are in the right place
        mu_n1, S_n = update_mu_S(mu_n, S_n, lambda_, x, y_n1)
        M = COV.to_numpy()
        M[:, :] = S_n

        # ! DEBUG
        # M = M.astype(float)
        # if utilities.is_pos_def(M) != True:
        # print("Non positive definite matrix M")
        # raise Exception("Non positive definite matrix M")
        # if utilities.is_pos_semi_def(M) != True:
        # print("Non positive semidefinite matrix M")
        # raise Exception("Non positive semidefinite matrix M")

        idx = sorted(list(attributes["KG_mu"].keys()))
        for i in range(len(idx)):
            attributes["KG_mu"][idx[i]] = mu_n1[i][0]

    return E_obj_value


def multisolve_scenarios(attributes, subproblem, scenarios):
    """Solves optimization problem for every scenario applied to the given subproblem.

    Returns
    ----------
    k : list
    """
    k = attributes["pool"].map(
        partial(
            optimize_subproblem,
            attributes["network"],
            attributes["crash_time"],
            attributes["crash_cost"],
            subproblem,
            attributes["t_init"],
            attributes["t_final"],
            attributes["pessimistic"],
            attributes["penalty_type"],
            attributes["m"],
            attributes["b1"],
            attributes["penalty_steps"],
        ),
        scenarios,
    )

    # ! Debug single threaded code
    # k=[]
    # for scenario in scenarios:
    # k.append(optimize_subproblem(attributes['network'],
    # attributes['crash_time'],
    # attributes['crash_cost'],
    # subproblem,
    # attributes['t_init'],
    # attributes['t_final'],
    # attributes['pessimistic'],
    # attributes['penalty_type'],
    # attributes['m'],
    # attributes['b1'],
    # attributes['penalty_steps'],
    # scenario))

    return k


def bootstrap_E_STD(subproblem, attributes):
    """Evaluates a variance-reduction bootstrap method.

    Returns
    ----------
    Zmean : float
       Mean, as obtained from boostrap methd
    Zstd : float
       Std, as obtained from boostrap methd
    """
    # Get the samples from subproblem
    subproblem_sample_sol = np.array(subproblem["sample_solutions"])

    b = attributes["resamples"]
    alpha = attributes["confidence"]

    # Use bootstrap to approximate mean
    boost_mean_dist = bs.bootstrap(
        values=subproblem_sample_sol,
        stat_func=bs_stats.mean,
        alpha=alpha,
        num_iterations=b,
        iteration_batch_size=None,
        is_pivotal=True,
        num_threads=1,
        return_distribution=True,
    )

    # Use bootstrap to approximate std
    boost_std_dist = bs.bootstrap(
        values=subproblem_sample_sol,
        stat_func=bs_stats.std,
        alpha=alpha,
        num_iterations=b,
        iteration_batch_size=None,
        is_pivotal=True,
        num_threads=1,
        return_distribution=True,
    )

    return boost_mean_dist.mean(), boost_std_dist.mean()


def bootstrap_KG(subproblem, attributes, sample_solutions):
    """Evaluates bootstrap for KG samples.

    Returns
    ----------
    KGmean : float
       Mean, as obtained from boostrap methd
    KGstd : float
       Std, as obtained from boostrap methd
    """
    # Get the KG samples
    # KG_sample_sol = np.array(subproblem['KG_sample_sol'])
    KG_sample_sol = np.array(sample_solutions)

    b = attributes["resamples"]
    alpha = attributes["confidence"]

    # Use bootstrap to approximate mean
    boost_mean_dist = bs.bootstrap(
        values=KG_sample_sol,
        stat_func=bs_stats.mean,
        alpha=alpha,
        num_iterations=b,
        iteration_batch_size=None,
        is_pivotal=True,
        num_threads=1,
        return_distribution=True,
    )

    # Use bootstrap to approximate std
    boost_std_dist = bs.bootstrap(
        values=KG_sample_sol,
        stat_func=bs_stats.std,
        alpha=alpha,
        num_iterations=b,
        iteration_batch_size=None,
        is_pivotal=True,
        num_threads=1,
        return_distribution=True,
    )

    return boost_mean_dist.mean(), boost_std_dist.mean()


def assign_scenarios(partition_list, total_scenarios, method):
    """Computes and assigns scenarios to nodes of partition_list.

    This is done on Random, Random_1, and Distance methods.

    Return
    ----------
    assigned_scenarios_dict : dictionary
       Dictionary with suproblem-index as key and value the assigned
        scenario to the subproblem.
    """
    elements = len(partition_list)
    scenarios_length = len(total_scenarios)
    assigned_scenarios_dict = {}

    if method == "Random":
        # Randomly assign scenarios to elements of partition_list
        assigned_scenarios = np.random.choice(elements, scenarios_length)

    elif method == "Random_1":
        # Assign all scenarios to a single randomly
        # chosen subproblem of partition_list
        assigned_scenarios = np.array(
            [np.random.randint(0, len(partition_list))] * scenarios_length
        )

    elif method == "Distance":
        # Obtain random distribution of samples based
        # on distance of its corresponding (Z_E, Z_std)
        probabilities = rank_by_distance(partition_list)
        assigned_scenarios = np.random.choice(
            elements, scenarios_length, p=probabilities
        )

    for i, subproblem in enumerate(partition_list):
        scenarios = []
        scenarios_to_consider = np.where(assigned_scenarios == i)[0]
        scenarios = [total_scenarios[j].tolist() for j in scenarios_to_consider]
        assigned_scenarios_dict[i] = scenarios

    return assigned_scenarios_dict


def assign_scenarios_KG(partition_list, total_scenarios, attributes):
    """Compute and assign scenarios to nodes of partition_list on KG method Parameter.

    Return
    ----------
    assigned_scenarios_dict : dictionary
       Dictionary with suproblem-index as key and value
        the assigned scenario to the subproblem.
    """
    global COV

    # scenarios_length = len(total_scenarios)
    assigned_scenarios_dict = {}

    # Call KG algorith to select next index to sample
    S = COV.to_numpy(dtype=float)
    mu = np.fromiter(attributes["KG_mu"].values(), dtype=float)
    lambda_ = np.fromiter(attributes["KG_lambda"].values(), dtype=float)

    L = KG_multi(mu, S, lambda_, attributes["pool"])

    # Convert solution to selected subproblem name
    selected_name = sorted(list(attributes["KG_mu"].keys()))[L[0]]

    # Assign all scenarios to the selected index
    # assigned_scenarios = np.array([selected_name] * scenarios_length)

    # for i, subproblem in enumerate(partition_list):
    # scenarios = []
    # scenarios_to_consider = np.where(assigned_scenarios == i)[0]
    # scenarios = [total_scenarios[j].tolist() for j in scenarios_to_consider]
    # assigned_scenarios_dict[i] = scenarios

    for i, subproblem in enumerate(partition_list):
        if subproblem["name"] == selected_name:
            assigned_scenarios_dict[i] = [
                total_scenarios[j].tolist() for j in range(len(total_scenarios))
            ]
        else:
            assigned_scenarios_dict[i] = []

    return assigned_scenarios_dict


def assign_scenarios_pareto(partition_list, total_scenarios, beta, method):
    """Compute and assign scenarios to nodes of partition_list.

    This is done on Pareto Inverse and Pareto_Boltzman methods.

    Return
    ----------
    assigned_scenarios_dict : dictionary
       Dictionary with suproblem-index as key and value
        the assigned scenario to the subproblem.
    """
    elements = len(partition_list)
    scenarios_length = len(total_scenarios)
    assigned_scenarios_dict = {}

    # List to hold Pareto fronts
    front_list = []

    if method == "Pareto_Inverse" or method == "Pareto_Boltzman":
        # Rank subproblem based pareto method
        num_fronts, front_list, probabilities = pareto_fronts_probabilities(
            partition_list, beta, method
        )

        # Distribute scenarios using the obtained Pareto front probability
        assigned_scenarios = np.random.choice(
            elements, scenarios_length, p=probabilities
        )

    for i, subproblem in enumerate(partition_list):
        scenarios = []
        scenarios_to_consider = np.where(assigned_scenarios == i)[0]
        scenarios = [total_scenarios[j].tolist() for j in scenarios_to_consider]
        assigned_scenarios_dict[i] = scenarios

    return assigned_scenarios_dict


def rank_by_distance(partition_list):
    """Ranks nodes in partition list by distance of its (Z_E, Z_std) point.

    Returns
    ----------
    probabilities : list
       List of proabilities ranking nodes in partition_list
    """
    distance = []
    probabilities = []
    for subproblem in partition_list:
        if subproblem["Z_E"] == np.inf:
            distance.append(np.inf)
        else:
            distance.append(
                math.pow(
                    (
                        math.pow(subproblem["Z_E"], 2)
                        + math.pow(subproblem["Z_std"] - 1, 2)
                    ),
                    0.5,
                )
            )

    # Obtain probablities based on distance
    for dist in distance:
        probabilities.append(dist / np.sum(distance))

    return probabilities


def pareto_fronts_probabilities(partition_list, beta, probability_method):
    """Get non-dominated fronts and its probabilities.

    Returns
    ----------
    num_fronts : int
       Number of non-dominated fronts
    ndr : numpy.ndarray
       Non-domination ranks
    probability : dictionary
       Dictionary with subproblem as keys and assigned probability values.
    """
    # Get (-|Z_E|, -|Z_std|) points for all subproblems
    points = [
        [-(abs(subproblem["Z_E"])), -(abs(1 - abs(subproblem["Z_std"])))]
        for subproblem in partition_list
    ]

    # Obtain the non-dominated sorting of points
    ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(points=points)

    num_fronts = len(ndf)

    # Calculate probability at for all fronts.
    front_prob = {}

    if probability_method == "Pareto_Inverse":
        # Calculating denominator
        denominator = 0
        for i in range(num_fronts):
            denominator = denominator + len(ndf[i]) * ((1 / (i + 1)) ** beta)

        # Calculating probability
        for i in range(num_fronts):
            front_prob[i] = ((1 / (i + 1)) ** beta) / denominator

    elif probability_method == "Pareto_Boltzman":
        # Calculating denominator
        denominator = 0
        for i in range(num_fronts):
            denominator = denominator + len(ndf[i]) * (np.e ** (-beta * (i + 1)))

        # Calculating probability
        for i in range(num_fronts):
            front_prob[i] = (np.e ** (-beta * (i + 1))) / denominator

    # Calculate probability for all elements of partition list
    prob_dict = {}
    for i in range(len(ndf)):
        for j in range(len(ndf[i])):
            prob_dict[ndf[i][j]] = front_prob[i]

    probabilities = [i for (j, i) in sorted(prob_dict.items())]

    return num_fronts, ndf, probabilities


def update_row_col(subproblem1, subproblem2, partition_list, attributes):
    """Updates the last two rows and columns of COV.

    The update is done with the exponential kernel values of the string
    distances of the corresponding subproblem constraints and finds the
    nearest covariance matrix.
    """
    global COV

    kg_sigma = attributes["KG_sigma"]
    kg_l = attributes["KG_l"]
    s1_const = subproblem1["constraints"]
    s2_const = subproblem2["constraints"]

    const_vec = [
        partition_list[i]["constraints"] for i in range(len(partition_list) - 2)
    ]

    # Obtain constraints as strings
    s1_const_str = ""
    for k in s1_const.keys():
        s1_const_str += str(s1_const[k])

    s2_const_str = ""
    for k in s2_const.keys():
        s2_const_str += str(s2_const[k])

    const_str = []
    for d in const_vec:
        d_str = ""
        for k in d.keys():
            d_str += str(d[k])
        const_str.append(d_str)

    const_str += [s1_const_str, s2_const_str]

    # Calculate 1st column to fill
    dist1 = np.array([l_dist(s1_const_str, string) for string in const_str])
    col1 = kg_sigma**2 * np.exp(-np.square(dist1) / (2 * kg_l**2))
    # col1[len(col1)-2] += 2*kg_l**2
    # col1[len(col1)-2] += kg_sigma**2 #positive-definite adjustment: max variance

    # Calculate 1st column to fill
    dist2 = np.array([l_dist(s2_const_str, string) for string in const_str])
    col2 = kg_sigma**2 * np.exp(-np.square(dist2) / (2 * kg_l**2))
    # col2[len(col2)-1] += 2*kg_l**2
    # col2[len(col2)-1] += kg_sigma**2 #positive-definite adjustment: max variance

    # Update COV last two rows and columns
    M = COV.to_numpy()
    k = M.shape[0] - 1

    M[:, k - 1] = col1
    M[k - 1, :] = col1
    M[:, k] = col2
    M[k, :] = col2

    M = M.astype(float)
    M = cov_nearest(M, threshold=1e-6, n_fact=10000, return_all=False)
    S = COV.to_numpy()
    S[:, :] = M

    # ! DEBUG
    # if utilities.is_pos_def(M) is not True:
    # print("Non positive definite matrix M")
    # raise Exception("Non positive definite matrix M")
    # if utilities.is_pos_semi_def(M) is not True:
    # print("Non positive semidefinite matrix M")
    # raise Exception("Non positive semidefinite matrix M")

    return
