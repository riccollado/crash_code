import gurobipy as gp


def uncrashed_project_time(network, scenario):
    """Solves the uncrashed unpenalized scheduling problem on a single scenario

    Parameters
    ----------
    network : networkx.classes.digraph.DiGraph
       Network digraph
    scenario : list
       Single scenario of activity times

    Returns
    ----------
    ObjVal : float
       Objective value of solution
    OptSol : list
       Optimal solution
    """

    no_of_nodes = network.number_of_nodes()

    # Setting up the optimization problem
    model = gp.Model("BoundOptimizationProblem")
    model.params.OutputFlag = 0  # Supress logging
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
            # Debug
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
