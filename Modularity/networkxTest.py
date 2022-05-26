"""


"""

import networkx as nx
import random
from math import floor
from math import log
from main import createNetwork as createAdjacency
from main import graphDictionary as adjacencyToDict
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


def randomWalkUntilCycle(G):
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
    previousNodes = []
    if random.random() > .5:
        head = x
        tail = y
        previousNodes = [tail]
    else:
        head = y
        tail = x
        previousNodes = [tail]
    while head not in previousNodes:
        neighbors = list(G[head])
        neighbors.remove(tail)
        if len(neighbors) == 0:
            print("Dead end")
            return None
        previousNodes.append(head)
        head, tail = random.choice(neighbors), head
    start = previousNodes.index(head)
    cycle = previousNodes[start:]
    return cycle


def RNBRW(G, n):
    for _ in range(n):
        cycle = randomWalkUntilCycle(G)
        if cycle is not None:
            G[cycle[0]][cycle[-1]]['rnbrw_weight'] += 1
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


# testAdjacency()
# testDictionary()

def main(n):
    # Create community graph
    G = communityBuilder(n, 5, .7, .2)

    # RNBRW(G, len(G.edges))
    RNBRW(G, floor(n))
    print("rnbrw", nx.algorithms.community.louvain_communities(G, 'rnbrw_weight', 1))
    print("cycle", nx.algorithms.community.louvain_communities(G, 'cycle_rnbrw', 1))

    # rnbrw = [edge for edge in G.edges() if G[edge[0]][edge[1]]['rnbrw_weight'] > 0]
    # rnbrw = [edge for edge in G.edges() if G[edge[0]][edge[1]]['cycle_rnbrw'] > 1]
    # nx.draw(G, pos=nx.circular_layout(G))
    # nx.draw_networkx_nodes(G, pos=nx.circular_layout(G))
    # nx.draw_networkx_edges(G, pos=nx.circular_layout(G), edgelist=rnbrw, edge_color='r')
    # plt.show()


main(500)




