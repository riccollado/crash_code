"""Generate experiment networks."""

import io
import secrets
from itertools import product
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
from numpy import floor


SECURE_RANDOM = secrets.SystemRandom()


def network_skeleton(
    num_nodes: int,
    num_layers: int,
) -> List[Tuple[int, int]]:
    """Generate edge list of network skeleton.

    Parameters
    ----------
    num_nodes : int
        Number of nodes for the network.
    num_layers : int
        Number of layers in the network skeleton.

    Returns
    -------
    edges : List[Tuple[int, int]]
        List of pairs of nodes comprising the skeleton edges.
    """
    # Generate initial layer partition
    layer_partition = [0, num_nodes - 1, num_nodes - 2]
    layer_partition.extend(
        SECURE_RANDOM.sample(range(1, num_nodes - 2), num_layers - 3)
    )
    layer_partition.sort()

    # Populate layers
    layers = [[0]]
    for i in range(1, num_layers):
        layers.append(list(range(layer_partition[i - 1] + 1, layer_partition[i] + 1)))

    # Edge list
    edges = []

    # Add edges from a layer to the next
    for i in range(len(layers) - 1):
        a_layer = layers[i]
        b_layer = layers[i + 1]

        SECURE_RANDOM.shuffle(a_layer)
        SECURE_RANDOM.shuffle(b_layer)

        # We have two cases depending on which layer is larger
        if len(a_layer) <= len(b_layer):
            a_layer_list = [[a] for a in a_layer]
            index = [0, len(b_layer)]
            index.extend(SECURE_RANDOM.sample(range(1, len(b_layer)), len(a_layer) - 1))
            index.sort()
            b_layer_list = [
                b_layer[index[i] : index[i + 1]] for i in range(len(index) - 1)
            ]

            # Add edges
            for j in range(len(a_layer)):
                edges.extend(list(product(a_layer_list[j], b_layer_list[j])))

        else:
            b_layer_list = [[b] for b in b_layer]
            index = [0, len(a_layer)]
            index.extend(SECURE_RANDOM.sample(range(1, len(a_layer)), len(b_layer) - 1))
            index.sort()
            a_layer_list = [
                a_layer[index[i] : index[i + 1]] for i in range(len(index) - 1)
            ]

            # Add edges
            for j in range(len(b_layer)):
                edges.extend(list(product(a_layer_list[j], b_layer_list[j])))

    return edges


def generate_network(
    num_nodes: int,
    num_layers: int,
    density: float,
) -> Tuple[nx.DiGraph, bytes, Dict[int, Tuple[float, float]]]:
    """Generate a connected network graph.

    Parameters
    ----------
    num_nodes : int
        Number of nodes for the network.
    num_layers : int
        Number of layers in the network skeleton.
    density : float
        Desired network density (0 < density < 1).

    Returns
    -------
    G : networkx.DiGraph
        Network digraph.
    binary_figure : bytes
        Binary representation of the network figure in PDF format.
    pos : dict
        Dictionary with node positions for plotting.
    """
    # Generate a skeleton graph We assume that the nodes in the graph are ALWAYS a full
    # range of numbers
    nodes = list(range(num_nodes))
    edges = network_skeleton(num_nodes, num_layers)
    network_graph = nx.DiGraph()
    network_graph.add_nodes_from(nodes)
    network_graph.add_edges_from(edges)

    # Obtain pyplot layout from skeleton graph
    pos = nx.nx_pydot.pydot_layout(network_graph, prog="dot")

    # Generate possible edges to add
    nodes = sorted(network_graph.nodes())
    pairs = [
        (x, y)
        for x, y in product(nodes, nodes)
        if x < y
        and x != 0
        and y != len(nodes) - 1
        and (x, y) not in network_graph.edges()
    ]

    # Calculate required number of nodes to reach density but only if new density > old
    # density and not exceeding the available pairs
    n = min(
        max(
            int(
                floor(
                    (
                        (density - 2 * nx.density(network_graph))
                        * (
                            network_graph.number_of_nodes() ** 2
                            - network_graph.number_of_nodes()
                        )
                    )
                    / 2
                )
            ),
            0,
        ),
        len(pairs),
    )

    # Add new edges
    network_graph.add_edges_from(SECURE_RANDOM.sample(pairs, n))

    # Plot figure to io stream and store binary values
    figure = io.BytesIO()
    nx.draw(
        network_graph,
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

    return network_graph, binary_figure, pos
