"""
David Rollo's code for testing NetworkX


"""

import networkx as nx
import random
from math import floor, log2
import matplotlib.pyplot as plt
from sklearn.metrics import normalized_mutual_info_score, adjusted_mutual_info_score
import numpy as np
import csv
import os
import directedlouvain


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
    initial = -t
    divisor = len(G.edges) * initial
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
    initial = -t
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
    initial = -t
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


"""
def digraphLabeling(G):
    storage = [[] for node in range(len(G.nodes) + 1)]
    for node in range(1, len(G.nodes)):
        for edge in list(G[node]):
            storage[edge].append(node)
    for node in range(1, len(storage)):
        G.nodes[node]['in_edge'] = storage[node]
"""


def directedRandomWalkUntilCycle(G, head, tail, retrace=False):
    """
    Parameters
    ----------
    G : directed graph as an input
    head : the node from which the directed edge is coming from
    tail : the node which the directed edge is going to
    retrace: whether or not we retrace reciprocal edges

    Returns
    -------
    boolean : whether the random walk encountered a retracing edge
    retraced_edge : the two nodes which the directed edge that forms a cycle in the order of the directed edge
    """
    path_walked = {tail}
    while head not in path_walked:
        neighbors = list(G.neighbors(head))
        # print(neighbors)
        if tail in neighbors and not retrace:
            neighbors.remove(tail)
        if len(neighbors) == 0:
            return False, head, tail
        path_walked.add(tail)
        tail = head
        head = random.choice(neighbors)
    return True, head, tail


def directedRandomWalkUntilCycle2(G, head, tail):
    """
    Parameters
    ----------
    G : directed graph as an input
    head : the node from which the directed edge is coming from
    tail : the node which the directed edge is going to
    retrace: whether or not we retrace reciprocal edges

    Returns
    -------
    boolean : whether the random walk encountered a retracing edge
    cycle : the two nodes which the directed edge that forms a cycle in the order of the directed edge
    """
    path_walked = []
    while head not in path_walked:
        neighbors = list(G.successors(head))
        if tail in neighbors:
            neighbors.remove(tail)
        if len(neighbors) == 0:
            return False, path_walked
        path_walked.append(head)
        tail, head = head, random.choice(neighbors)
    start = path_walked.index(head)
    cycle = path_walked[start:]
    return True, cycle


def DRNBRW(G, t):
    """
    Parameters
    ----------
    G : directed graph as input
    t : number of iterations through the edges (t * m iterations)

    Returns true when complete
    """
    initial = .01
    # divisor = len(G.edges) * initial
    nx.set_edge_attributes(G, values=initial, name='directed_rnbrw')
    # failed = 0
    for head, tail in G.edges:
        for trial in range(t):
            complete, head, tail = directedRandomWalkUntilCycle(G, head, tail)
            if complete:
                G[tail][head]['directed_rnbrw'] += 1
                #divisor += 1
    #for head, tail in G.edges:
        #G[head][tail]['directed_rnbrw'] /= divisor
    # print("directed RNBRW", failed)
    return True


def backtrackDRW(G, t):
    """
    Parameters
    ----------
    G : directed graph as input
    t : number of iterations through the edges (t * m iterations)

    Returns true when complete
    """
    initial = .01
    # divisor = len(G.edges) * initial
    nx.set_edge_attributes(G, values=initial, name='backtrack')
    failed = 0
    for head, tail in G.edges:
        for trial in range(t):
            complete, head, tail = directedRandomWalkUntilCycle(G, head, tail, True)
            if complete:
                G[tail][head]['backtrack'] += 1
                # divisor += 1
            else:
                failed += 1
    # for head, tail in G.edges:
        # G[head][tail]['backtrack'] /= divisor
    # print("retraced RNBRW", failed)
    # print(len(G.edges))
    return True


def directedCycle(G, t=1):
    """
    Parameters
    ----------
    G : directed graph as input
    t : number of iterations through the edges (t * m iterations)

    Returns true when complete
    """
    initial = .01
    nx.set_edge_attributes(G, values=initial, name='directed_cycle')
    for head, tail in G.edges:
        for trial in range(t):
            complete, cycle = directedRandomWalkUntilCycle2(G, head, tail)
            if complete:
                for node in range(len(cycle)):
                    G[cycle[node-1]][cycle[node]]['directed_cycle'] += 1
    return True


def randomWalkUntilCycleZigZag(G, head, tail, direction=-1):
    """
    Beginning with directed graph G, we choose a random edge in G.
    Since G is directed, we randomly decide a head and tail for the edge.
    We then randomly walk, without backtracking until we revisit a node
    OR we visit a node with no edges (besides backtracking).

    We return the cycle found by the random walk where the retracing edge connects [0] and [-1]
    """
    path = {tail}
    # Random walk until cycle
    while head not in path:
        if direction < 0:
            neighbors = list(G.successors(head))
        else:
            neighbors = list(G.predecessors(head))
        if tail in neighbors:
            neighbors.remove(tail)
        if head in neighbors:
            neighbors.remove(head)
        if len(neighbors) == 0:
            return False, head, tail, direction
        path.add(head)
        head, tail = random.choice(neighbors), head
        direction *= -1
    # Return statements (retraced edge or cycle list)
    return True, head, tail, direction


def ZRNBRW(G, t=1):
    # Update the graph edge attributes for each edge found in a cycle
    initial = .01
    #divisor = len(G.edges) * initial
    nx.set_edge_attributes(G, values=initial, name='zigzag')
    for head, tail in G.edges:
        for trial in range(t):
            completed, head, tail, d = randomWalkUntilCycleZigZag(G, head, tail)
            if completed:
                if d > 0:
                    G[tail][head]['zigzag'] += 1
                else:
                    G[head][tail]['zigzag'] += 1
                # divisor += 1
    # for head, tail in G.edges:
        # G[head][tail]['zigzag'] /= divisor
    # print("zigzag", failed)
    return True


def ZCNBRW(G, t=1):
    # Update the graph edge attributes for each edge found in a cycle
    initial = .01
    # divisor = len(G.edges) * initial
    nx.set_edge_attributes(G, values=initial, name='zigzag_cycle')
    for head, tail in G.edges:
        for trial in range(t):
            completed, cycle, d = randomWalkUntilCycleZigZag2(G, head, tail)
            if completed:
                zigzagMethod(G, cycle, d, 'zigzag_cycle')
            completed, cycle, d = randomWalkUntilCycleZigZag2(G, tail, head, direction=-1)
            if completed:
                zigzagMethod(G, cycle, d, 'zigzag_cycle')
    # for head, tail in G.edges:
        # G[head][tail]['zigzag_cycle'] /= divisor
    # print("zigzag_cycle", divisor)
    return True


def zigzagMethod(G, cycle, d, string):
    if d < 0:
        G[cycle[-1]][cycle[0]][string] += 1
    else:
        G[cycle[0]][cycle[-1]][string] += 1
    offset = (d+1) // 2
    if len(cycle) % 2 == 0:
        for node in range(0, len(cycle), 2):
            if node < len(cycle) - 2:
                G[cycle[node+1-offset]][cycle[node+offset]][string] += 1
                G[cycle[node+1+offset]][cycle[node+2-offset]][string] += 1
            elif node == len(cycle) - 2:
                G[cycle[node+1-offset]][cycle[node+offset]][string] += 1
    else:
        for node in range(0, len(cycle), 2):
            if node < len(cycle) - 2:
                G[cycle[node+offset]][cycle[node+1-offset]][string] += 1
                G[cycle[node+2-offset]][cycle[node+1+offset]][string] += 1
            elif node == len(cycle) - 2:
                G[cycle[node+1]][cycle[node]][string] += 1
    return len(cycle)


def randomWalkUntilCycleZigZag2(G, head, tail, direction=1):
    """
    Beginning with directed graph G, we choose a random edge in G.
    Since G is directed, we randomly decide a head and tail for the edge.
    We then randomly walk, without backtracking until we revisit a node
    OR we visit a node with no edges (besides backtracking).

    We return the cycle found by the random walk where the retracing edge connects [0] and [-1]
    """
    path = [tail]
    # Random walk until cycle
    while head not in path:
        direction *= -1
        if direction < 0:
            neighbors = list(G.successors(head))
        else:
            neighbors = list(G.predecessors(head))
        if tail in neighbors:
            neighbors.remove(tail)
        if head in neighbors:
            neighbors.remove(head)
        if len(neighbors) == 0:
            return False, path, direction
        path.append(head)
        head, tail = random.choice(neighbors), head
    # Return statements (retraced edge or cycle list)
    start = path.index(head)
    cycle = path[start:]
    return True, cycle, direction


def randomWalkCousin(G, tail, head, direction=1):
    """
    Beginning with directed graph G, we choose a random edge in G.
    Since G is directed, we randomly decide a head and tail for the edge.
    We then randomly walk, without backtracking until we revisit a node
    OR we visit a node with no edges (besides backtracking).

    We return the cycle found by the random walk where the retracing edge connects [0] and [-1]
    """
    path = [tail]
    edges = [[tail, head]]
    # Random walk until cycle
    while head not in path:
        direction *= -1
        if direction > 0:
            neighbors = list(G.successors(head))
        else:
            neighbors = list(G.predecessors(head))
        if tail in neighbors:
            neighbors.remove(tail)
        if head in neighbors:
            neighbors.remove(head)
        if len(neighbors) == 0:
            return False, path, direction
        path.append(head)
        head, tail = random.choice(neighbors), head
        if direction > 0:
            edges.append([tail, head])
        else:
            edges.append([head, tail])
    # Return statements (retraced edge or cycle list)
    start = path.index(head)
    cycleNodes = path[start:]
    cycleEdges = edges[start:]
    print(cycleEdges)
    return True, cycleNodes, direction


def weightedZCNBRW(G, t=1):
    # Update the graph edge attributes for each edge found in a cycle
    initial = .01
    # divisor = len(G.edges) * initial
    nx.set_edge_attributes(G, values=initial, name='weighted_zigzag')
    for head, tail in G.edges:
        for trial in range(t):
            completed, cycle, d = randomWalkUntilCycleZigZag2(G, head, tail)
            if completed:
                if d < 0:
                    G[cycle[-1]][cycle[0]]['weighted_zigzag'] += 1 / len(cycle)
                else:
                    G[cycle[0]][cycle[-1]]['weighted_zigzag'] += 1 / len(cycle)
                offset = (d+1) // 2
                if len(cycle) % 2 == 0:
                    for node in range(0, len(cycle), 2):
                        if node < len(cycle) - 2:
                            G[cycle[node+1-offset]][cycle[node+offset]]['weighted_zigzag'] += 1 / len(cycle)
                            G[cycle[node+1+offset]][cycle[node+2-offset]]['weighted_zigzag'] += 1 / len(cycle)
                        elif node == len(cycle) - 2:
                            G[cycle[node+1-offset]][cycle[node+offset]]['weighted_zigzag'] += 1 / len(cycle)
                else:
                    for node in range(0, len(cycle), 2):
                        if node < len(cycle) - 2:
                            G[cycle[node+offset]][cycle[node+1-offset]]['weighted_zigzag'] += 1 / len(cycle)
                            G[cycle[node+2-offset]][cycle[node+1+offset]]['weighted_zigzag'] += 1 / len(cycle)
                        elif node == len(cycle) - 2:
                            G[cycle[node+1]][cycle[node]]['weighted_zigzag'] += 1 / len(cycle)
                # divisor += 1
    # for head, tail in G.edges:
        # G[head][tail]['weighted_zigzag'] /= divisor
    # print("weighted_zigzag", divisor)
    return True


def directedGraphToCSV(G, file, extra='', t=1):
    inCommunityLabel(G)
    # unweightedGroups = nx.algorithms.community.louvain_communities(G, seed=100)
    DRNBRW(G, t=t)
    backtrackDRW(G, t=t)
    # drnbrwGroups = nx.algorithms.community.louvain_communities(G, 'rnbrw', seed=100)
    ZRNBRW(G, t=t)
    ZCNBRW(G, t=t)
    weightedZCNBRW(G, t=t)
    # zigzagGroups = nx.algorithms.community.louvain_communities(G, 'cycle', seed=100)
    # graphNodesToCSV(G, file)
    directedGraphEdgesToCSV(G, file, t, extra=extra)


def directedGraphEdgesToCSV(G, file, t, extra=''):
    string = ""
    if len(extra) > 0:
        string = "csvEdgesDirected" + "/" + file + extra + "_" + str(t) + "m.csv"
    else:
        string = "csvEdgesDirected" + "/" + file + "_" + str(t) + "m.csv"
    if not os.access(string, 0):
        print("Error: Failed to access CSV file")
    with open(string, 'w') as csvfile:
        fieldnames = ['in_comm', 'directed_rnbrw', 'backtrack', 'zigzag', 'zigzag_cycle', 'weighted_zigzag']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for head, tail in G.edges:
            writer.writerow({'in_comm': G[head][tail]['in_comm'],
                             'directed_rnbrw': G[head][tail]['directed_rnbrw'],
                             'backtrack': G[head][tail]['backtrack'],
                             'zigzag': G[head][tail]['zigzag'],
                             'zigzag_cycle': G[head][tail]['zigzag_cycle'],
                             'weighted_zigzag': G[head][tail]['weighted_zigzag']})


def subgraphRNBRW(G):
    source, target = random.choice(list(G.edges))
    complete, cycle = directedRandomWalkUntilCycle2(G, source, target)
    H = G.subgraph(cycle)
    # nx.set_edge_attributes(H, values='Red', name='color')
    pos = nx.spring_layout(G, k=0.15, iterations=20,seed=1234)
    plt.figure(15, figsize=(60, 60))
    nx.draw(G, pos)
    plt.show()


def visualize(G, truecom):
    color_map = []
    for node in G:
        for i in range(len(truecom)):
            if node in truecom[i]:
                color_map.append(i)

    pos = nx.spring_layout(G, k=0.15, iterations=20,seed=1234)
    plt.figure(15, figsize=(60,60))
    nx.draw(G, pos, node_size=200, arrowsize=10, node_color=color_map, cmap=plt.cm.hsv)
    plt.show()


def reciprocityIndex(G):
    N = len(G.nodes) * (len(G.nodes) - 1)
    totalL, mutualL = countReciprocals(G)
    if N == totalL:
        return 1
    return (mutualL * N - (totalL * totalL)) / (totalL * N - (totalL * totalL))


def countReciprocals(G):
    totalL = len(G.edges)
    mutualL = 0
    for tail, head in G.edges:
        if tail in G[head]:
            mutualL += 1
    return totalL, mutualL


"""
def countReciprocalsWeighted(G):
    totalL = 0
    mutualL = 0
    for tail, head, weight in G.edges:
        totalL += weight
        if tail in G[head]:
            mutualL += weight
    return totalL, mutualL


def reciprocalRatioWeighted(G):
    totalL = 0
    mutualL = 0
    for tail, head, weight in G.edges:
        totalL += weight
        if tail in G[head]:
            mutualL += weight
    return mutualL / totalL
"""


def reciprocalRatio(G):
    totalL = len(G.edges)
    mutualL = 0
    for tail, head in G.edges:
        if tail in G[head]:
            mutualL += 1
    return mutualL / totalL


def communityReciprocalRatio(G, communities, weighted=False):
    # I'm going to assume the communities are iterable
    reciprocityIndexList = []
    total = 0
    for comm in communities:
        # comm should be a collection of node names
        if len(comm) <= 1:
            # How do we want to deal with singletons or pairs?
            # It is probably dishonest to rate a singleton as having 100% reciprocity
            # At the same time, 0% is unfair
            # Pairs of two can't backtrack
            # For now these cases will not be counted
            continue
        H = G.subgraph(comm)
        if weighted:
            reciprocityIndexList.append(reciprocalRatio(H) * len(comm))
            total += len(comm)
        else:
            reciprocityIndexList.append(reciprocalRatio(H))
            total += 1
    return sum(reciprocityIndexList)/total


def communityReciprocity(G, communities, weighted=False):
    # I'm going to assume the communities are iterable
    reciprocityIndexList = []
    total = 0
    for comm in communities:
        if len(comm) <= 1:
            # How do we want to deal with singletons or pairs?
            # It is probably dishonest to rate a singleton as having 100% reciprocity
            # At the same time, 0% is unfair
            # Pairs of two can't backtrack
            # For now these cases will not be counted
            continue
        H = G.subgraph(comm)
        if weighted:
            reciprocityIndexList.append(reciprocityIndex(H) * len(comm))
            total += len(comm)
        else:
            reciprocityIndexList.append(reciprocityIndex(H))
            total += 1
    return sum(reciprocityIndexList)/total


def finalTest(G, communities, initialWeight, latex=True):
    """
    Coordinate between the graph and the communities
    - initialWeight
    - |S|
    - |C|
    - max(C)
    - r
    - rho
    """
    singleCount = 0
    maxSize = 0
    total = 0
    count = 1
    initialWeight = float(initialWeight[:1] + '.' + initialWeight[1:])
    for comm in communities:
        total += len(comm)
        count += 1
        if len(comm) == 1:
            singleCount += 1
        if len(comm) > maxSize:
            maxSize = len(comm)
    aveComm = round(total / count, 3)
    r = str(round(communityReciprocalRatio(G, communities, weighted=True), 3))
    rho = str(round(communityReciprocity(G, communities, weighted=True), 3))
    # What is the weight?
    data = [initialWeight, str(singleCount), str(len(communities)-singleCount), str(maxSize), str(aveComm), r, rho]
    if latex:
        return ' & '.join(data)
    return data


def communityCounter(communities):
    counter = {}
    for comm in communities:
        val = floor(log2(len(comm)))
        if val in counter:
            counter[val] += 1
        else:
            counter[val] = 1
    return counter









