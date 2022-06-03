"""



"""

from networkxTest import *
# from graphData import *
from time import time
from math import log


def main(n, group_count=25, draw=False):
    # Create community graph
    print(n)
    start = time()
    # G = communityBuilder(n, group_count, p_in=.7, p_out=.2)
    G = LFRBenchmark(n)
    # IDENTIFY PRE-BUILT COMMUNITIES
    communityList = identifyLFRCommunities(G)
    print("construction time", time() - start)
    print("modularity", modularity(G, communityList))

    # CLASSIFY COMMUNITIES
    start = time()
    RNBRW(G, len(G.edges))
    rnbrw = nx.algorithms.community.louvain_communities(G, 'rnbrw', 1)
    print("RNBRW time m", time() - start)
    print("Modularity", modularity(G, rnbrw))

    start = time()
    CNBRW(G, n)
    cycle = nx.algorithms.community.louvain_communities(G, 'cycle', 1)
    print("Cycle time n", time() - start)
    print("Modularity", modularity(G, cycle))

    start = time()
    CNBRW(G, 2 * n)
    cycle = nx.algorithms.community.louvain_communities(G, 'cycle', 1)
    print("Cycle time 2n", time() - start)
    print("Modularity", modularity(G, cycle))

    start = time()
    CNBRW(G, floor(n * log(n)))
    cycle = nx.algorithms.community.louvain_communities(G, 'cycle', 1)
    print("Cycle time nlogn", time() - start)
    print("Modularity", modularity(G, cycle))

    start = time()
    CNBRW(G, len(G.edges))
    cycle = nx.algorithms.community.louvain_communities(G, 'cycle', 1)
    print("Cycle time m", time() - start)
    print("Modularity", modularity(G, cycle))

    if draw:
        rnbrw = [edge for edge in G.edges() if G[edge[0]][edge[1]]['rnbrw'] > 0]
        # rnbrw = [edge for edge in G.edges() if G[edge[0]][edge[1]]['cycle'] > 1]
        # nx.draw(G, pos=nx.circular_layout(G))
        nx.draw_networkx_nodes(G, pos=nx.circular_layout(G))
        nx.draw_networkx_edges(G, pos=nx.circular_layout(G), edgelist=rnbrw, edge_color='r')
        plt.show()


def mainRetraceStudy(n):
    print("\n", n)
    start = time()
    G = LFRBenchmark(n)
    # G = nx.barbell_graph(4, 0)

    # IDENTIFY PRE-BUILT COMMUNITIES
    communityList = identifyLFRCommunities(G)
    print("construction time", time() - start)

    # CLASSIFY COMMUNITIES unweighted
    print("\n UNWEIGHTED")
    start = time()
    unweightedGroups = nx.algorithms.community.louvain_communities(G)
    print("Control time", time() - start)
    print("Modularity", nx.algorithms.community.modularity(G, unweightedGroups))
    print("NMI", NMI(n, communityList, unweightedGroups))
    print("adjNMI", adjustNMI(n, communityList, unweightedGroups))

    # CLASSIFY COMMUNITIES by RNBRW
    print("\n RNBRW")
    start = time()
    RNBRW(G, len(G.edges))
    rnbrwGroups = nx.algorithms.community.louvain_communities(G, 'rnbrw')
    print("RNBRW time m", time() - start)
    print("Modularity", nx.algorithms.community.modularity(G, rnbrwGroups))
    print("NMI", NMI(n, communityList, rnbrwGroups))
    print("adjNMI", adjustNMI(n, communityList, rnbrwGroups))

    # CLASSIFY COMMUNITIES by Cycle
    print("\n CYCLE")
    start = time()
    CNBRW(G, len(G.nodes))
    cycleGroups = nx.algorithms.community.louvain_communities(G, 'cycle')
    print("CNBRW time n", time() - start)
    print("Modularity", nx.algorithms.community.modularity(G, cycleGroups))
    print("NMI", NMI(n, communityList, cycleGroups))
    print("adjNMI", adjustNMI(n, communityList, cycleGroups))


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
