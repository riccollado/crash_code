"""Uncrashed bounds method."""

from typing import List, Tuple

import gurobipy as gp
import networkx as nx


def uncrashed_project_time(
    network: nx.DiGraph,
    scenario: List[float],
) -> Tuple[float, List[int]]:
    """Solves the uncrashed unpenalized scheduling problem on a single scenario.

    Parameters
    ----------
    network : networkx.DiGraph
        Network digraph.
    scenario : list of float
        Single scenario of activity times.

    Returns
    -------
    obj_val : float
        Objective value of solution.
    opt_sol : list of int
        Optimal solution.
    """
    no_of_nodes = network.number_of_nodes()

    # Setting up the optimization problem
    model = gp.Model("BoundOptimizationProblem")
    model.params.OutputFlag = 0  # Suppress logging
    model.params.Thread = 0  # Automatic multithread execution of solver

    # Activities start-time variables (integer)
    s = {}
    for i in range(no_of_nodes):
        s[i] = model.addVar(vtype=gp.GRB.INTEGER, name="s" + str(i))

    # Update model to integrate new variables
    model.update()

    # Network flow model constraints
    constr = {}
    for node in network.nodes:
        successors = network.successors(node)
        for succ in successors:
            # ! DEBUG
            # print("Adding constraint for {}->{}".format(node, succ))
            constr[(node, succ)] = model.addConstr(
                s[succ] >= s[node] + scenario[node], name="constr" + str(succ)
            )

    # Optimization function
    model.setObjective(s[no_of_nodes - 1], gp.GRB.MINIMIZE)
    model.optimize()

    # Retrieve  solution
    ObjVal = model.getAttr("ObjVal")
    OptSol = [int(s[i].X) for i in range(no_of_nodes)]

    return (ObjVal, OptSol)
