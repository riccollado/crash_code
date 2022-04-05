"""Generate subproblems."""

import gurobipy as gp

from generator_penalty import (
    generate_penalty_vals_exponential,
    generate_penalty_vals_linear,
)


def optimize_subproblem(
    network,
    crash_time,
    crash_cost,
    subproblem,
    t_init,
    t_final,
    pessimistic,
    penalty_type,
    m,
    b1,
    penalty_steps,
    scenario,
):
    """Defines and solve an optimization problem where some variables are fixed.

    Parameters
    ----------
    network : networkx.classes.digraph.DiGraph
       Full network diagraph

    crash_time : list
       Activirt crash percentage

    crash_cost : list
       Actitvity crash cost

    subproblem : dict
       Dictionary with keys: 'variables', 'constraints',
       'recordset', 'singleton', 'sample_solutions', 'E',
       'num_sample_solutions', 'total_scenarios_done', and
       'scenarios_done'

    t_init : float
       Lower threshold for penalty function

    t_final : float
       Upper threshold for penalty function

    outlocation : str
       Name of output files folder

    pessimistic : list
       Duration of pessimistic values of activities

    penalty_type : str
       Either "linear" or "exponential"

    m : float
       M parameter for penalty function

    b1 : float
       b1 parameter for penalty function

    scenario : list
       List of values for all nodes of the graph (including
       start and end) that comprises one scenario

    penalty_steps : __type__
        Steps used in penalty function.

    Returns
    ----------
    scenariodetailslog : dict
       Dictionary containing details on the solution of the
       subpoblem. The kays are: 'Penalty', 'Objective func Value',
       'Regular Costs', 'Project Duration', 'Schedule',
       'Penalty Variables', and all optimal crash values.
    """
    # Get number of nodes
    no_of_nodes = network.number_of_nodes()

    # Defnite intervals of penalty step function
    d = (t_final - t_init) / penalty_steps
    t = [t_init + (j * d) for j in range(int(penalty_steps) + 1)]

    # Setting up the optimization problem
    model = gp.Model("MainOptimizationProblem")
    model.params.OutputFlag = 0  # Supress logging
    model.params.Thread = 0  # Automatic multithread execution of solver

    # Obtain problem variables
    variables = subproblem["variables"]

    # Crashing activities binary variables
    x = {}
    for i in range(len(variables)):
        x[i] = model.addVar(vtype=gp.GRB.BINARY, name="x" + str(i))

    # Activities schedule start-time variables (integer)
    s = {}
    for i in range(no_of_nodes):
        s[i] = model.addVar(vtype=gp.GRB.INTEGER, name="s" + str(i))

    # Handle for final activity
    s_f = s[no_of_nodes - 1]

    # Penalty function variables ()
    z = {}
    for i in range(1, int(penalty_steps) + 1):
        z[i] = model.addVar(vtype=gp.GRB.BINARY, name="z" + str(i))

    # Update model to integrate new variables
    model.update()

    # Add crashing constraints set in sub-problem while partitioning
    constraints = subproblem["constraints"]
    crash_constr = {}
    for i, j in constraints.items():
        crash_constr[i] = model.addConstr(x[i] == j, name="c" + str(i))

    # Network flow model constraints
    flow_constr = {}
    for node in network.nodes:
        successors = network.successors(node)
        for succ in successors:
            flow_constr[(node, succ)] = model.addConstr(
                s[succ]
                >= s[node]
                + scenario[node]
                - x[node] * crash_time[node] * scenario[node],
                name="flow_constr" + str(succ),
            )

    # Penalty costs
    if penalty_type == "linear":
        Lambda = generate_penalty_vals_linear(t, m, b1)
    else:  # i.e. penalty_type==exponential
        Lambda = generate_penalty_vals_exponential(t, m, b1)

    # BigM calculation
    BigM = 4 * sum(pessimistic)

    # Penalty-related constraints
    penalty_minus_constr = {}
    penalty_plus_constr = {}
    for i in range(1, int(penalty_steps) + 1):
        penalty_minus_constr[i] = model.addConstr(
            s_f >= t[i] - BigM * (1 - z[i]), name="p-" + str(i)
        )
        penalty_plus_constr[i] = model.addConstr(
            s_f <= t[i] + BigM * z[i], name="p+" + str(i)
        )

    # Objective function
    model.setObjective(
        gp.quicksum(Lambda[i] * z[i] for i in range(1, int(penalty_steps) + 1))
        + gp.quicksum(x[i] * crash_cost[i] for i in range(no_of_nodes)),
        gp.GRB.MINIMIZE,
    )

    # Solve the model
    model.optimize()

    # Dictionary to return model solution
    scenariodetailslog = {}
    scenariodetailslog["Penalty"] = sum(
        [
            Lambda[i] * model.getVarByName("z" + str(i)).X
            for i in range(1, int(penalty_steps) + 1)
        ]
    )
    scenariodetailslog["Objective func Value"] = model.getAttr("ObjVal")
    scenariodetailslog["Regular Costs"] = (
        scenariodetailslog["Objective func Value"] - scenariodetailslog["Penalty"]
    )
    scenariodetailslog["Project Duration"] = model.getVarByName(
        "s" + str(no_of_nodes - 1)
    ).X
    scenariodetailslog["Schedule"] = [
        model.getVarByName("s" + str(i)).X for i in range(no_of_nodes)
    ]
    scenariodetailslog["Penalty Variables"] = [
        model.getVarByName("z" + str(i)).X for i in range(1, int(penalty_steps) + 1)
    ]

    for i in range(no_of_nodes):
        scenariodetailslog["x" + str(i)] = model.getVarByName("x" + str(i)).X

    return scenariodetailslog
