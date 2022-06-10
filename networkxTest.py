"""
David Rollo's code for testing NetworkX


"""

import networkx as nx
import random
from math import floor
from graphData import createAdjacency
from graphData import adjacencyToDict
import matplotlib.pyplot as plt
from sklearn.metrics import normalized_mutual_info_score, adjusted_mutual_info_score
import numpy as np
import csv
import os


def testAdjacency():
    """
    Create and draw a graph based on an adjacency matrix
    """
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
    """
    A test to create a graph from a dictionary
    """
    A = adjacencyToDict(createAdjacency(10, .4))

    G = nx.Graph()
    G.add_nodes_from(list(A))
    for node, edgeList in A.items():
        for edge in edgeList:
            G.add_edge(node, edge)

    nx.draw(G, with_labels=True, font_weight='bold')
    plt.show()


def createGraph(n, p):
    """
    A method which returns a graph with n nodes
    Every edge has a probability p to be created
    """
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
    """
    head, tail = random.choice(list(G.edges))
    # Choose a direction
    if random.random() > .5:
        head, tail = tail, head
    path = {tail}
    # Random walk until cycle
    while head not in path:
        neighbors = list(G[head])
        neighbors.remove(tail)
        if head in neighbors:
            neighbors.remove(head)
        if len(neighbors) == 0:
            return False, tail, head
        path.add(head)
        head, tail = random.choice(neighbors), head
    # Return retraced edge
    return True, tail, head


def RNBRW(G, t):
    #
    initial = .01
    divisor = len(G.edges) * initial
    nx.set_edge_attributes(G, values=initial, name='rnbrw')
    for head, tail in G.edges:
        for trial in range(t):
            complete, one, two = randomWalkFromEdge(G, head, tail)
            if complete:
                G[one][two]['rnbrw'] += 1
                divisor += 1
            complete, one, two = randomWalkFromEdge(G, tail, head)
            if complete:
                G[one][two]['rnbrw'] += 1
                divisor += 1
    for head, tail in G.edges:
        G[head][tail]['rnbrw'] /= divisor
    return True


def randomWalkFromEdge(G, head, tail):
    """
    Beginning with input graph G, we choose a random edge in G.
    Since G is undirected, we randomly decide a head and tail for the edge.
    We then randomly walk, without backtracking until we revisit a node
    OR we visit a node with no edges (besides backtracking).
    """
    path = {tail}
    # Random walk until cycle
    while head not in path:
        neighbors = list(G[head])
        neighbors.remove(tail)
        if head in neighbors:
            neighbors.remove(head)
        if len(neighbors) == 0:
            return False, tail, head
        path.add(head)
        head, tail = random.choice(neighbors), head
    # Return retraced edge
    return True, tail, head


def randomWalkUntilCycle2(G, head, tail):
    """
    Beginning with input graph G, we choose a random edge in G.
    Since G is undirected, we randomly decide a head and tail for the edge.
    We then randomly walk, without backtracking until we revisit a node
    OR we visit a node with no edges (besides backtracking).

    We return the cycle found by the random walk where the retracing edge connects [0] and [-1]
    """
    path = [tail]
    # Random walk until cycle
    while head not in path:
        neighbors = list(G[head])
        neighbors.remove(tail)
        if head in neighbors:
            neighbors.remove(head)
        if len(neighbors) == 0:
            return False, path
        path.append(head)
        head, tail = random.choice(neighbors), head
    # Return statements (retraced edge or cycle list)
    start = path.index(head)
    cycle = path[start:]
    return True, cycle


def CNBRW(G, t=1):
    # Update the graph edge attributes for each edge found in a cycle
    initial = .01
    divisor = 0
    nx.set_edge_attributes(G, values=initial, name='cycle')
    for head, tail in G.edges:
        for trial in range(t):
            completed, cycle = randomWalkUntilCycle2(G, head, tail)
            if completed:
                for node in range(len(cycle)):
                    G[cycle[node]][cycle[node-1]]['cycle'] += 1
                    divisor += 1
            completed, cycle = randomWalkUntilCycle2(G, tail, head)
            if completed:
                for node in range(len(cycle)):
                    G[cycle[node]][cycle[node-1]]['cycle'] += 1
                    divisor += 1
    for head, tail in G.edges:
        G[head][tail]['cycle'] /= divisor
    return True


def weightedCNBRW(G, t=1):
    """
    Update the graph edge attributes for each edge found in a cycle
    Update by each edge by reciprocal cycle length
    """
    initial = .01
    divisor = len(G.edges) * initial
    nx.set_edge_attributes(G, values=initial, name='weightedCycle')
    for head, tail in G.edges:
        for trial in range(t):
            completed, cycle = randomWalkUntilCycle2(G, head, tail)
            if completed:
                for node in range(len(cycle)):
                    G[cycle[node]][cycle[node-1]]['weightedCycle'] += 1 / len(cycle)
                divisor += 1
            completed, cycle = randomWalkUntilCycle2(G, tail, head)
            if completed:
                for node in range(len(cycle)):
                    G[cycle[node]][cycle[node-1]]['weightedCycle'] += 1 / len(cycle)
                divisor += 1
    for head, tail in G.edges:
        G[head][tail]['weightedCycle'] /= divisor
    return True


def hybridRNBRW(G, t=1):
    """
    Update the graph edge attributes for each edge found in a cycle
    Update by each edge by reciprocal cycle length
    Retracing edge is also given a bonus weight
    """
    initial = .01
    divisor = len(G.edges) * initial
    nx.set_edge_attributes(G, values=initial, name='hybrid')
    for head, tail in G.edges:
        for trial in range(t):
            completed, cycle = randomWalkUntilCycle2(G, head, tail)
            if completed:
                for node in range(len(cycle)):
                    G[cycle[node]][cycle[node-1]]['hybrid'] += 1 / len(cycle)
                G[cycle[0]][cycle[-1]]['hybrid'] += 1
                divisor += 2
            completed, cycle = randomWalkUntilCycle2(G, tail, head)
            if completed:
                for node in range(len(cycle)):
                    G[cycle[node]][cycle[node-1]]['hybrid'] += 1 / len(cycle)
                G[cycle[0]][cycle[-1]]['hybrid'] += 1
                divisor += 2
    for head, tail in G.edges:
        G[head][tail]['hybrid'] /= divisor
    return True


def communityBuilder(nodes, group_count, p_in, p_out):
    """
    Create a graph with nodes evenly divided between group_count
    p_in is probability of edge within group
    p_out is probability of edge across groups
    Returns the networkx graph object
    """
    G = nx.Graph()
    groups = []
    # Add connections within groups
    for i in range(0, nodes, nodes // group_count):
        start = i
        end = min(start + nodes // group_count, nodes)
        groupNodes = [val for val in range(start, end)]
        groups.append(groupNodes)
        community = set(groupNodes)
        for node in groupNodes:
            G.add_node(node, community=community)
        for node in range(len(groupNodes)):
            for edge in range(node+1, len(groupNodes)):
                if random.random() < p_in:
                    G.add_edge(groupNodes[node], groupNodes[edge], rnbrw=0, cycle=0)
    # Add connections between groups
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            for edge in range(floor(len(groups[i]) * len(groups[j]) * p_out)):
                G.add_edge(random.choice(groups[i]), random.choice(groups[j]), rnbrw=0, cycle=0)
    return G


def LFRBenchmark(n, tau1=2.5, tau2=1.5, average_degree=7.0, mu=.1, min_degree=None, max_degree=None, min_community=30, max_community=70):
    """ !!! only min_degree XOR average_degree must be specified, otherwise a NetworkXError is raised. !!!
    Benchmark test to determine how well an algorithm is at community detection.
    Returns networkx graph object
    """
    if min_degree is not None:
        average_degree = None
    if max_degree is None:
        max_degree = n
    # Initialize graph
    G = None
    try:
        G = nx.generators.community.LFR_benchmark_graph(n=n, tau1=tau1, tau2=tau2, average_degree=average_degree, mu=mu,
            min_degree=min_degree, max_degree=max_degree, min_community=min_community, max_community=max_community)
    except nx.ExceededMaxIterations:
        return G
    G.remove_edges_from(nx.selfloop_edges(G))
    return G


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


def identifyLFRCommunities(G):
    # IDENTIFY PRE-BUILT COMMUNITIES
    return list({frozenset(G.nodes[v]["community"]) for v in G})


def NMI(n, trueGroups, testGroups):
    return normalized_mutual_info_score(groupsToList(n, trueGroups), groupsToList(n, testGroups))


def adjustNMI(n, trueGroups, testGroups):
    return adjusted_mutual_info_score(groupsToList(n, trueGroups), groupsToList(n, testGroups))


def groupsToList(n, communities):
    # Index i stores the ith node's community
    groupFormat = [0 for _ in range(n)]
    for group in range(len(communities)):
        for val in communities[group]:
            groupFormat[int(val)] = group
    return groupFormat


def inCommunityLabel(G):
    nx.set_edge_attributes(G, values=False, name="in_comm")
    for head, tail in G.edges:
        if G.nodes[head]['community'] == G.nodes[tail]['community']:
            G[head][tail]['in_comm'] = True
    return G


def graphNodesToCSV(G, file):
    string = "csvNodes" + "/" + file + ".csv"
    f = open(string, "w")
    with open(string, newline='') as csvfile:
        fieldnames = ['community', '']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    writer.writerow({'first_name': 'Baked', 'last_name': 'Beans'})
    writer.writerow({'first_name': 'Lovely', 'last_name': 'Spam'})
    writer.writerow({'first_name': 'Wonderful', 'last_name': 'Spam'})

    nx.write_weighted_edgelist(G, string)


def runAllWeightings():
    print()


def GraphToCSV(G, file, t=1):
    inCommunityLabel(G)
    unweightedGroups = nx.algorithms.community.louvain_communities(G, seed=100)
    RNBRW(G, t=t)
    rnbrwGroups = nx.algorithms.community.louvain_communities(G, 'rnbrw', seed=100)
    CNBRW(G, t=t)
    cycleGroups = nx.algorithms.community.louvain_communities(G, 'cycle', seed=100)
    weightedCNBRW(G, t=t)
    weightedCycleGroups = nx.algorithms.community.louvain_communities(G, 'weightedCycle', seed=100)
    hybridRNBRW(G, t=t)
    hybridGroups = nx.algorithms.community.louvain_communities(G, 'hybrid', seed=100)
    # graphNodesToCSV(G, file)
    graphEdgesToCSV(G, file, t)


def graphEdgesToCSV(G, file, t):
    string = "csvEdges" + "/" + file + "_" + str(t) +"m.csv"
    if not os.access(string, 0):
        print("Error: Failed to access CSV file")
    with open(string, 'w') as csvfile:
        fieldnames = ['in_comm', 'rnbrw', 'cycle', 'weightedCycle', 'hybrid']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for head, tail in G.edges:
            writer.writerow({'in_comm': G[head][tail]['in_comm'],
                             'rnbrw': G[head][tail]['rnbrw'],
                             'cycle': G[head][tail]['cycle'],
                             'weightedCycle': G[head][tail]['weightedCycle'],
                             'hybrid': G[head][tail]['hybrid']})
