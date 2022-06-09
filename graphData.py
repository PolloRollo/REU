""""
David Rollo's code testing graph creation methods
"""

import random


def createAdjacency(n, p):
    network = [[0 for _ in range(n)] for __ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            if random.random() < p:
                network[i][j] = 1
                network[j][i] = 1
    return network


def degreeCount(network):
    return [sum(node) for node in network]


# Designed to measure the "connective-ness" of a network
# Meant to be optimized for community detection
def modularity(G, communities):
    """
    Given a graph G and a list of communities calculate and return the modularity
    -.5 <= M <= 1
    """
    m = 2 * len(G.edges)
    modularityVal = 0
    n = len(G.nodes)
    for group in communities:
        for i in group:
            for j in range(n):
                if j in group:  # Dirac delta
                    A = 0
                    if j in G.adj[i]:  # If there is an edge
                        A = 1
                    modularityVal += A - (len(G[i]) * len(G[j]) / m)

    modularityVal /= m
    return modularityVal


def adjacencyToDict(adjacency):
    graphDict = {}
    for i in range(len(adjacency)):
        graphDict[i] = []
        for j in range(len(adjacency[i])):
            if adjacency[i][j] > 0:
                graphDict[i].append(j)
    return graphDict


def main():
    network = createAdjacency(10, .5)
    graph = adjacencyToDict(network)
    for i, j in graph.items():
        print(i, j)
    modularityFunction(network)


# main()
