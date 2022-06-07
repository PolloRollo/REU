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


def RNBRW(G, n):
    # Update the graph edge attributes for each retraced edge found
    i = 0
    nx.set_edge_attributes(G, values=0.01, name='rnbrw')
    while i < n:
        completed_cycle, tail, head = randomWalkUntilCycle(G)
        if completed_cycle:
            G[tail][head]['rnbrw'] += 1
            i += 1
    return True


def equalRNBRW(G):
    n = len(G.nodes)
    nx.set_edge_attributes(G, values=.01, name='equal')
    for head, tail in G.edges:
        for trial in range(5):
            complete, one, two = randomWalkFromEdge(G, head, tail)
            if complete:
                G[one][two]['equal'] += 1
            complete, one, two = randomWalkFromEdge(G, tail, head)
            if complete:
                G[one][two]['equal'] += 1
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


def randomWalkUntilCycle2(G, cycle=False):
    """
    Beginning with input graph G, we choose a random edge in G.
    Since G is undirected, we randomly decide a head and tail for the edge.
    We then randomly walk, without backtracking until we revisit a node
    OR we visit a node with no edges (besides backtracking).

    We return the cycle found by the random walk where the retracing edge connects [0] and [-1]
    """
    head, tail = random.choice(list(G.edges))

    # Choose a direction
    if random.random() > .5:
        head, tail = tail, head
    path = [tail]
    # Random walk until cycle
    while head not in path:
        neighbors = list(G[head])
        neighbors.remove(tail)
        if head in neighbors:
            neighbors.remove(head)
        if len(neighbors) == 0:
            return False, cycle
        path.append(head)
        head, tail = random.choice(neighbors), head
    # Return statements (retraced edge or cycle list)
    start = path.index(head)
    cycle = path[start:]
    return True, cycle


def CNBRW(G, n):
    # Update the graph edge attributes for each edge found in a cycle
    i = 0
    initial = .01
    delta = 1
    nx.set_edge_attributes(G, values=initial, name='cycle')
    while i < n:
        completed, cycle = randomWalkUntilCycle2(G, cycle=True)
        if completed:
            # print([G.nodes[i]['community'] for i in cycle])
            for node in range(len(cycle)):
                G[cycle[node]][cycle[node-1]]['cycle'] += delta
            i += 1
    return True


def weightedCNBRW(G, n):
    """
    Update the graph edge attributes for each edge found in a cycle
    Update by each edge by reciprocal cycle length
    """
    i = 0
    nx.set_edge_attributes(G, values=.01, name='weightedCycle')
    while i < n:
        completed, cycle = randomWalkUntilCycle2(G)
        if completed:
            for node in range(len(cycle)):
                G[cycle[node]][cycle[node-1]]['weightedCycle'] += 1/len(cycle)
            i += 1
    return True


def cycleStudy(G, n):
    """
    Test how often cycles occur completely within communities or across communities
    Depends on LFR benchmark labels for communities
    """
    inGroupCycleLength = {}
    outGroupCycleLength = {}
    inGroupCount = 0
    outGroupCount = 0
    for _ in range(n):
        cycle = randomWalkUntilCycle(G, cycle=True)
        group = G.nodes[cycle[0]]['community']
        inGroup = True
        for node in cycle:
            if node not in group:
                inGroup = False
                outGroupCount += 1
                if len(cycle) in outGroupCycleLength:
                    outGroupCycleLength[len(cycle)] += 1
                else:
                    outGroupCycleLength[len(cycle)] = 1
                break
        if inGroup:
            inGroupCount += 1
            if len(cycle) in inGroupCycleLength:
                inGroupCycleLength[len(cycle)] += 1
            else:
                inGroupCycleLength[len(cycle)] = 1
    print("inGroupCount", inGroupCount)
    print("outGroupCount", outGroupCount)
    return [inGroupCycleLength, outGroupCycleLength]


def retraceStudy(G, n):
    """
    Test how often retraced edges occur completely within communities or across communities
    Depends on LFR benchmark labels for communities
    """
    inGroupCount = 0
    outGroupCount = 0
    for _ in range(n):
        renewal = randomWalkUntilCycle(G)
        if G.nodes[renewal[0]] == G.nodes[renewal[1]]:
            inGroupCount += 1
        else:
            outGroupCount += 1
    print("inGroupCount", inGroupCount)
    print("outGroupCount", outGroupCount)
    return [inGroupCount, outGroupCount]


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
