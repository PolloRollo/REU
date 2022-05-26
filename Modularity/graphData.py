""""




"""

import random
import networkx
import networkxTest


def createNetwork(n, p):
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
def modularityFunction(network):
    d = degreeCount(network)
    m = sum(d)
    modularity = 0
    n = len(network)
    for i in range(n):
        for j in range(n):
            modularity = network[i][j] - (d[i] * d[j] / m) * (1)  # Change to community Dirac delta
    modularity /= m
    print(modularity)


def graphDictionary(adjacency):
    graphDict = {}
    for i in range(len(adjacency)):
        graphDict[i] = []
        for j in range(len(adjacency[i])):
            if adjacency[i][j] > 0:
                graphDict[i].append(j)
    return graphDict


def main():
    network = createNetwork(10, .5)
    graph = graphDictionary(network)
    for i, j in graph.items():
        print(i, j)
    modularityFunction(network)