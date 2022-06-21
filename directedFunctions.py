import networkx as nx
import random
import os


# METHODS FORE READING GRAPH FILES
def readDiGraph(file):
    """ Returns networkx DiGraph from generated network file """
    if not os.access(file, 0):
        print("Error: Failed to access file")
    f = open(file, "r")
    G = nx.DiGraph()

    for line in f:
        source, target = line.split()
        source = int(source)
        target = int(target)
        if source not in G.nodes:
            G.add_node(source)
        if target not in G.nodes:
            G.add_node(target)
        G.add_edge(source, target)

    f.close()

    return G


def readDiCommunities(file):
    """ Returns list of communities from generated file """
    if not os.access(file, 0):
        print("Error: Failed to access file")
    f = open(file, "r")

    communities = dict()

    for line in f:
        node, community = line.split()
        node = int(node)
        community = int(community)
        if community not in communities.keys():
            communities.update({community: set()})
        communities[community].add(node)

    f.close()

    return list(communities.values())


def readDiAll(file):
    if str(file[-4:]) != ".dat":
        file += ".dat"
    G = readDiGraph("digraphs/networks/"+file)
    groups = readDiCommunities("digraphs/communities/"+file)

    for group in groups:
        community = group
        for node in group:
            G.add_node(int(node), community=community)
    return G


# RANDOM WALK BASED METHODS FOR WEIGHTING GRAPHS
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
    nx.set_edge_attributes(G, values=initial, name='directed_rnbrw')
    for head, tail in G.edges:
        for trial in range(t):
            complete, head, tail = directedRandomWalkUntilCycle(G, head, tail)
            if complete:
                G[tail][head]['directed_rnbrw'] += 1
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
    nx.set_edge_attributes(G, values=initial, name='backtrack')
    failed = 0
    for head, tail in G.edges:
        for trial in range(t):
            complete, head, tail = directedRandomWalkUntilCycle(G, head, tail, True)
            if complete:
                G[tail][head]['backtrack'] += 1
            else:
                failed += 1
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
            neighbors = list(G[head])
        else:
            neighbors = list(G.nodes[head]['in_edge'])
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
    nx.set_edge_attributes(G, values=initial, name='zigzag')
    for head, tail in G.edges:
        for trial in range(t):
            completed, head, tail, d = randomWalkUntilCycleZigZag(G, head, tail)
            if completed:
                if d > 0:
                    G[tail][head]['zigzag'] += 1
                else:
                    G[head][tail]['zigzag'] += 1
    return True


def ZCNBRW(G, t=1):
    # Update the graph edge attributes for each edge found in a cycle
    initial = .01
    nx.set_edge_attributes(G, values=initial, name='zigzag_cycle')
    for head, tail in G.edges:
        for trial in range(t):
            completed, cycle, d = randomWalkUntilCycleZigZag2(G, head, tail)
            if completed:
                zigzagMethod(G, cycle, d, 'zigzag_cycle')
            completed, cycle, d = randomWalkUntilCycleZigZag2(G, tail, head, direction=-1)
            if completed:
                zigzagMethod(G, cycle, d, 'zigzag_cycle')
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
            neighbors = list(G[head])
        else:
            neighbors = list(G.nodes[head]['in_edge'])
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


def weightedZCNBRW(G, t=1):
    # Update the graph edge attributes for each edge found in a cycle
    initial = .01
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
    return True


def digraphLabeling(G):
    storage = [[] for node in range(len(G.nodes) + 1)]
    for node in range(1, len(G.nodes)):
        for edge in list(G[node]):
            storage[edge].append(node)
    for node in range(1, len(storage)):
        G.nodes[node]['in_edge'] = storage[node]


# RUNNING ALL FUNCTIONS AT ONCE

def returnFiles():
    files = ['1ln_500_1', '2ln_500_1', '3ln_500_1',
            '1ln_500_2', '2ln_500_2', '3ln_500_2',
            '1ln_500_3', '2ln_500_3', '3ln_500_3',
            '1ln_500_4', '2ln_500_4', '3ln_500_4',
            '1ln_1000_1', '2ln_1000_1', '3ln_1000_1',
            '1ln_1000_2', '2ln_1000_2', '3ln_1000_2',
            '1ln_1000_3', '2ln_1000_3', '3ln_1000_3',
            '1ln_1000_4', '2ln_1000_4', '3ln_1000_4',
            '1ln_5000_1', '2ln_5000_1', '3ln_5000_1',
            '1ln_5000_2', '2ln_5000_2', '3ln_5000_2',
            '1ln_5000_3', '2ln_5000_3', '3ln_5000_3',
            '1ln_5000_4', '2ln_5000_4', '3ln_5000_4',
            '1ln_10000_1', '2ln_10000_1', '3ln_10000_1',
            '1ln_10000_2', '2ln_10000_2', '3ln_10000_2',
            '1ln_10000_3', '2ln_10000_3', '3ln_10000_3',
            '1ln_10000_4', '2ln_10000_4', '3ln_10000_4']
    return files


def writeAllWeights(file, t=1):
    G = readDiAll(file)
    digraphLabeling(G)
    methods = ['directed_rnbrw', 'backtrack', 'zigzag', 'zigzag_cycle', 'weighted_zigzag', 'directed_cycle']
    DRNBRW(G, t)
    backtrackDRW(G, t)
    ZRNBRW(G, t)
    ZCNBRW(G, t)
    weightedZCNBRW(G, t)
    directedCycle(G, t)
    for method in methods:
        writeGraphWeights(G, folder="weightedDigraphs", file=file, method=method)


def createAllWeightFiles(t=1):
    files = returnFiles()
    for file in files:
        writeAllWeights(file, t=t)


def writeGraphWeights(G, folder, file, method, t=1):
    string = folder + "/" + method + "/" + file + "_" + str(t) + "m.txt"
    f = open(string, "w")
    for tail, head in G.edges:
        s = ' '.join([str(tail), str(head), str(G[tail][head][method]), "\n"])
        f.write(s)
