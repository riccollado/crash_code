import io
import random
from itertools import product

import matplotlib.pyplot as plt
import networkx as nx
from numpy import floor


def network_skeleton(num_nodes, num_layers):
    """Generates edge list of network skeleton

    Parameters
    ----------
    num_nodes : int
       Number of nodes for the network
    num_layers : int
       Number of layers in the network skeleton

    Returns
    ----------
    edges : list
       List of pairs of nodes comprising the skeleton edges
    """

    # Generate initial layer partition
    layer_partition = [0, num_nodes - 1, num_nodes - 2]
    layer_partition.extend(random.sample(range(1, num_nodes - 2), num_layers - 3))
    layer_partition.sort()

    # Populate layers
    layers = [[0]]
    for i in range(1, num_layers):
        layers.append(list(range(layer_partition[i - 1] + 1, layer_partition[i] + 1)))

    # Edge list
    edges = []

    # Add edges from a layer to the next
    for i in range(len(layers) - 1):
        A = layers[i]
        B = layers[i + 1]

        random.shuffle(A)
        random.shuffle(B)

        # We have two cases depending on which layer is larger
        if len(A) <= len(B):
            A = [[a] for a in A]
            index = [0, len(B)]
            index.extend(random.sample(range(1, len(B)), len(A) - 1))
            index.sort()
            B = [B[index[i] : index[i + 1]] for i in range(len(index) - 1)]

            # Add edges
            for j in range(len(A)):
                edges.extend(list(product(A[j], B[j])))

        else:
            B = [[b] for b in B]
            index = [0, len(A)]
            index.extend(random.sample(range(1, len(A)), len(B) - 1))
            index.sort()
            A = [A[index[i] : index[i + 1]] for i in range(len(index) - 1)]

            # Add edges
            for j in range(len(B)):
                edges.extend(list(product(A[j], B[j])))

    return edges


def generate_network(num_nodes, num_layers, density):
    """Generates a connected network graph

    Parameters
    ----------
    num_nodes : int
       Number of nodes for the network
    num_layers : int
       Number of layers in the network skeleton
    density: float
       Desired network density (0 < density < 1)

    Returns
    ----------
    G : networkx.classes.digraph.DiGraph
       Network digraph
    """
    # Generate a skeleton graph
    # We assume that the nodes in the graph
    # are ALWAYS a full range of numbers
    nodes = range(num_nodes)
    edges = network_skeleton(num_nodes, num_layers)
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    # Obtain pyplot layout from skeleton graph
    pos = nx.nx_pydot.pydot_layout(G, prog="dot")

    # Generate possible edges to add
    nodes = sorted(G.nodes())
    pairs = [
        (x, y)
        for x, y in product(nodes, nodes)
        if x < y and x != 0 and y != len(nodes) - 1 and (x, y) not in G.edges()
    ]

    # Calculate required number of nodes to reach density
    # but only if new density > old density
    # and not exceeding the available pairs
    n = min(
        max(
            int(
                floor(
                    (
                        (density - 2 * nx.density(G))
                        * (G.number_of_nodes() ** 2 - G.number_of_nodes())
                    )
                    / 2
                )
            ),
            0,
        ),
        len(pairs),
    )

    # Add new edges
    G.add_edges_from(random.sample(pairs, n))

    # Plot figure to io stream and store binary values
    figure = io.BytesIO()
    nx.draw(
        G,
        pos,
        with_labels=False,
        arrows=True,
        node_size=40,
        node_color="r",
        alpha=0.7,
        width=0.4,
    )
    plt.savefig(figure, format="pdf")
    binary_figure = figure.getvalue()
    figure.close()
    # plt.show()

    return G, binary_figure, pos
