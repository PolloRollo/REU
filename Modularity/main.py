"""



"""

from networkxTest import *
# from graphData import *
from time import time
import sklearn.metrics


def main(n, group_count=25, draw=False):
    # Create community graph
    print(n)
    start = time()
    # G = communityBuilder(n, group_count, p_in=.7, p_out=.2)
    G = LFRBenchmark(n)
    # IDENTIFY PRE-BUILT COMMUNITIES

    communityCount = 0
    communities = set()
    communityList = []
    for node in G.nodes:
        if node not in communities:
            communities |= G.nodes[node]['community']
            communityCount += 1
            communityList.append(G.nodes[node]['community'])
            # print(G.nodes[node]['community'])
    modValLFR = modularity(G, communityList)
    print("true group count", communityCount)
    print("construction time", time() - start)
    print(modValLFR)

    # CLASSIFY COMMUNITIES
    start = time()
    RNBRW(G, len(G.edges))
    rnbrw = nx.algorithms.community.louvain_communities(G, 'rnbrw', 1)
    # print("rnbrw", rnbrw)
    modValRnbrw = modularity(G, rnbrw)

    print("number of groups", len(rnbrw))
    print("RNBRW time m", time() - start)
    print(modValRnbrw)

    start = time()
    CNBRW(G, n)
    cycle = nx.algorithms.community.louvain_communities(G, 'cycle', 1)
    # print("cycle", cycle)
    modValCycle = modularity(G, cycle)

    print("number of groups", len(cycle))
    print("Cycle time n", time() - start)
    print(modValCycle)

    start = time()
    CNBRW(G, 2 * n)
    cycle = nx.algorithms.community.louvain_communities(G, 'cycle', 1)
    # print("cycle", cycle)
    modValCycle = modularity(G, cycle)
    print("number of groups", len(cycle))
    print("Cycle time 2n", time() - start)
    print(modValCycle)

    start = time()
    CNBRW(G, floor(n * log(n)))
    cycle = nx.algorithms.community.louvain_communities(G, 'cycle', 1)
    # print("cycle", cycle)
    modValCycle = modularity(G, cycle)
    print("number of groups", len(cycle))
    print("Cycle time nlogn", time() - start)
    print(modValCycle)

    start = time()
    CNBRW(G, len(G.edges))
    cycle = nx.algorithms.community.louvain_communities(G, 'cycle', 1)
    # print("cycle", cycle)
    modValCycle = modularity(G, cycle)
    print("number of groups", len(cycle))
    print("Cycle time m", time() - start)
    print(modValCycle)

    if draw:
        rnbrw = [edge for edge in G.edges() if G[edge[0]][edge[1]]['rnbrw'] > 0]
        # rnbrw = [edge for edge in G.edges() if G[edge[0]][edge[1]]['cycle'] > 1]
        # nx.draw(G, pos=nx.circular_layout(G))
        nx.draw_networkx_nodes(G, pos=nx.circular_layout(G))
        nx.draw_networkx_edges(G, pos=nx.circular_layout(G), edgelist=rnbrw, edge_color='r')
        plt.show()


def mainRetraceStudy(n):
    print("\n", n)
    # start = time()
    G = LFRBenchmark(n)
    # G = nx.barbell_graph(10, 2)

    """
    # IDENTIFY PRE-BUILT COMMUNITIES
    communityCount = 0
    communities = set()
    communityList = []
    for node in G.nodes:
        if node not in communities:
            communities |= G.nodes[node]['community']
            communityCount += 1
            communityList.append(G.nodes[node]['community'])
            # print(G.nodes[node]['community'])

    print("true group count", communityCount)
    # print("true group count", group_count)
    print("construction time", time() - start)
    """

    # CLASSIFY COMMUNITIES unweighted
    print("\n UNWEIGHTED")
    start = time()
    unweightedGroups = nx.algorithms.community.louvain_communities(G, 'unweighted', seed=123)
    # print("number of groups", len(unweightedGroups))
    modValUnweighted = nx.algorithms.community.modularity(G, unweightedGroups)
    print("Modularity", modValUnweighted)
    print("Control time", time() - start)

    # CLASSIFY COMMUNITIES by RNBRW
    print("\n RNBRW")
    start = time()
    RNBRW(G, len(G.edges))
    rnbrwGroups = nx.algorithms.community.louvain_communities(G, 'rnbrw', seed=123)
    # print("cycle", cycle)
    # print("number of groups", len(rnbrwGroups))
    modValRNBRW = nx.algorithms.community.modularity(G, rnbrwGroups)
    print("Modularity", modValRNBRW)
    print("RNBRW time m", time() - start)
    # print(G.adj)

    # CLASSIFY COMMUNITIES by Cycle
    print("\n CYCLE")
    start = time()
    CNBRW(G, 2 * len(G.nodes))
    cycleGroups = nx.algorithms.community.louvain_communities(G, 'cycle', seed=123)
    # print("number of groups", len(cycleGroups))
    modValCNBRW = nx.algorithms.community.modularity(G, cycleGroups)
    print("Modularity", modValCNBRW)
    print("CNBRW time 2 n", time() - start)


def mainCycleStudy(n):
    G = LFRBenchmark(n)
    cycleData = cycleStudy(G, 1000)
    print(cycleData)


def weightedCycleStudy(n):
    # Create community graph
    print(n)
    start = time()
    # G = communityBuilder(n, group_count, p_in=.7, p_out=.2)
    G = LFRBenchmark(n)
    # IDENTIFY PRE-BUILT COMMUNITIES

    communityCount = 0
    communities = set()
    communityList = []
    for node in G.nodes:
        if node not in communities:
            communities |= G.nodes[node]['community']
            communityCount += 1
            communityList.append(G.nodes[node]['community'])
            # print(G.nodes[node]['community'])

    print("true group count", communityCount)
    # print("true group count", group_count)
    print("construction time", time() - start)

    # CLASSIFY COMMUNITIES
    start = time()
    unweightedGroups = nx.algorithms.community.louvain_communities(G, 'unweighted', 1)
    # print("cycle", cycle)
    # print("number of groups", len(unweightedGroups))
    print("Control time n", time() - start)

    start = time()
    CNBRW(G, n)
    cycle = nx.algorithms.community.louvain_communities(G, 'cycle', 1)
    # print("cycle", cycle)
    print("number of groups", len(cycle))
    print("Cycle time n", time() - start)

    start = time()
    weightedCNBRW(G, n)
    weightedCycle = nx.algorithms.community.louvain_communities(G, 'cycle', 1)
    # print("cycle", cycle)
    print("number of groups", len(weightedCycle))
    print("Weighted Cycle time n", time() - start)


for i in range(4):
    # main(100 * 2**i)
    # mainCycleStudy(100 * 2**i)
    # weightedCycleStudy(100 * 2**i)
    mainRetraceStudy(1000 * 2**i)
