"""


"""

import networkx as nx
import random
from math import floor
from math import log
from graphData import createNetwork as createAdjacency
from graphData import graphDictionary as adjacencyToDict
import matplotlib.pyplot as plt


def testAdjacency():
    A = createAdjacency(10, .4)

    G = nx.Graph()
    n = len(A)
    G.add_nodes_from([_ for _ in range(n)])
    for node in range(n):
        for edge in range(n):
            if A[node][edge] > 0:
                G.add_edge(node, edge)

    nx.draw(G, with_labels=True, font_weight='bold')
    plt.show()


def testDictionary():
    A = adjacencyToDict(createAdjacency(10, .4))

    G = nx.Graph()
    G.add_nodes_from(list(A))
    for node, edgeList in A.items():
        for edge in edgeList:
            G.add_edge(node, edge)

    nx.draw(G, with_labels=True, font_weight='bold')
    plt.show()


def createGraph(n, p):
    A = createAdjacency(n, p)

    G = nx.Graph()
    n = len(A)
    G.add_nodes_from([_ for _ in range(n)])
    for node in range(n):
        for edge in range(n):
            if A[node][edge] > 0:
                G.add_edge(node, edge, rnbrw_weight=0, cycle_rnbrw=0)

    return G


def randomWalkUntilCycle(G, cycle=False):
    """
    Beginning with input graph G, we choose a random edge in G.
    Since G is undirected, we randomly decide a head and tail for the edge.
    We then randomly walk, without backtracking until we revisit a node
    OR we visit a node with no edges (besides backtracking).

    We return the cycle where the retracing edge connects [0] and [-1]
    -Bad coding practice to adjust the graph in this function?
    """
    x, y = random.choice(list(G.edges))
    head, tail = None, None
    path = []
    if random.random() > .5:
        head = x
        tail = y
        path = [tail]
    else:
        head = y
        tail = x
        path = [tail]
    while head not in path:
        neighbors = list(G[head])
        neighbors.remove(tail)
        if len(neighbors) == 0:
            print("Dead end")
            return None
        path.append(head)
        head, tail = random.choice(neighbors), head
    if cycle:
        start = path.index(head)
        cycle = path[start:]
        return cycle
    else:
        return [path[-1], head]


def RNBRW(G, n):
    for _ in range(n):
        renewal = randomWalkUntilCycle(G)
        if renewal is not None:
            G[renewal[0]][renewal[1]]['rnbrw_weight'] += 1


def CNBRW(G, n):
    for _ in range(n):
        cycle = randomWalkUntilCycle(G, cycle=True)
        if cycle is not None:
            for i in range(len(cycle)):
                G[cycle[i]][cycle[i-1]]['cycle_rnbrw'] += 1


def communityBuilder(nodes, group_count, p_in, p_out):
    G = nx.Graph()
    groups = []
    # Add connections within groups
    for i in range(0, nodes, nodes // group_count):
        start = i
        end = min(start + nodes // group_count, nodes)
        groupNodes = [val for val in range(start, end)]
        groups.append(groupNodes)
        G.add_nodes_from([_ for _ in groupNodes])
        for node in range(len(groupNodes)):
            for edge in range(node+1, len(groupNodes)):
                if random.random() < p_in:
                    G.add_edge(groupNodes[node], groupNodes[edge], rnbrw_weight=0, cycle_rnbrw=0)
    # Add connections between groups
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            for edge in range(floor(len(groups[i]) * len(groups[j]) * p_out)):
                G.add_edge(random.choice(groups[i]), random.choice(groups[j]), rnbrw_weight=0, cycle_rnbrw=0)
    return G


def LFRBenchmark(n, tau1, tau2, mu, average_degree, min_degree, max_degree, min_community, max_community, tol, iters):
    """
    Benchmark test to determine how well an algorithm is at community detection.

    Parameters
        n:      int - Number of nodes in the created graph.
        tau1:   float - Power law exponent for the degree distribution of the created graph. This value must be strictly greater than one.
        tau2:   float - Power law exponent for the community size distribution in the created graph. This value must be strictly greater than one.
        mu:     float - Fraction of inter-community edges incident to each node. This value must be in the interval [0, 1].
        average_degree: float - Desired average degree of nodes in the created graph. This value must be in the interval [0, n]. Exactly one of this and min_degree must be specified, otherwise a NetworkXError is raised.
        min_degree: int - Minimum degree of nodes in the created graph. This value must be in the interval [0, n]. Exactly one of this and average_degree must be specified, otherwise a NetworkXError is raised.
        max_degree: int - Maximum degree of nodes in the created graph. If not specified, this is set to n, the total number of nodes in the graph.
        min_community: int - Minimum size of communities in the graph. If not specified, this is set to min_degree.
        max_community: int - Maximum size of communities in the graph. If not specified, this is set to n, the total number of nodes in the graph.
        tol:    float - Tolerance when comparing floats, specifically when comparing average degree values.
        max_iters: int - Maximum number of iterations to try to create the community sizes, degree distribution, and community affiliations.

    Returns networkx graph object
        """
    return nx.generators.community.LFR_benchmark_graph(n, tau1, tau2, mu, average_degree,
                    min_degree, max_degree, min_community, max_community, tol, iters)

# testAdjacency()
# testDictionary()




