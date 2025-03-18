"""Uncrashed bounds method with Pyomo."""

from typing import List, Tuple

import networkx as nx
import pyomo.environ as pyo

# from pyomo.environ import (
#     ConcreteModel,
#     Var,
#     Objective,
#     Constraint,
#     SolverFactory,
#     NonNegativeIntegers,
#     minimize,
# )


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
    # Extracting network information
    no_of_nodes = network.number_of_nodes()
    nodes = pyo.Set(initialize=list(network.nodes))

    # Setting up the optimization problem
    model = pyo.ConcreteModel()

    # Activities start-time variables (integer)
    model.s = pyo.Var(nodes, domain=pyo.NonNegativeIntegers)

    # Network flow model constraints
    def network_constraints_rule(model, node, succ):
        return model.s[succ] >= model.s[node] + scenario[node]

    model.network_constraints = pyo.Constraint(
        [(node, succ) for node in network.nodes for succ in network.successors(node)],
        rule=network_constraints_rule,
    )

    # Optimization function
    model.obj = pyo.Objective(expr=model.s[no_of_nodes - 1], sense=pyo.minimize)

    # Solve the model
    solver = pyo.SolverFactory("glpk")
    solver.solve(model, tee=False)

    # Retrieve solution
    obj_val = model.obj()
    opt_sol = [int(model.s[i].value) for i in range(no_of_nodes)]

    return obj_val, opt_sol
