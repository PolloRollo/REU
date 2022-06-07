"""



"""

from networkxTest import *
# from graphData import *
from time import time
from math import log, sqrt, floor
import createGraphFiles


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


def mainRetraceStudy(n=1000):
    G = communityBuilder(n, floor(sqrt(n)), .75, .25)
    G.remove_edges_from(nx.selfloop_edges(G))
    # IDENTIFY PRE-BUILT COMMUNITIES
    # G = createGraphFiles.readAll("7_1000_4.txt")
    communityList = identifyLFRCommunities(G)

    # CLASSIFY COMMUNITIES unweighted
    print("\n UNWEIGHTED")
    start = time()
    unweightedGroups = nx.algorithms.community.louvain_communities(G, seed=100)
    print("Control time", time() - start)
    # print("Modularity", nx.algorithms.community.modularity(G, unweightedGroups))
    # print("NMI", NMI(n, communityList, unweightedGroups))
    print("adjNMI", adjustNMI(n, communityList, unweightedGroups))

    # CLASSIFY COMMUNITIES by Equal RNBRW
    print("\n EQUAL RNBRW")
    start = time()
    equalRNBRW(G)
    rnbrwGroups = nx.algorithms.community.louvain_communities(G, 'equal', seed=100)
    print("RNBRW time m", time() - start)
    #print("Modularity", nx.algorithms.community.modularity(G, rnbrwGroups))
    #print("NMI", NMI(n, communityList, rnbrwGroups))
    print("adjNMI", adjustNMI(n, communityList, rnbrwGroups))

    # CLASSIFY COMMUNITIES by RNBRW
    print("\n RNBRW")
    start = time()
    RNBRW(G, 2*len(G.edges))
    rnbrwGroups = nx.algorithms.community.louvain_communities(G, 'rnbrw', seed=100)
    print("RNBRW time m", time() - start)
    #print("Modularity", nx.algorithms.community.modularity(G, rnbrwGroups))
    #print("NMI", NMI(n, communityList, rnbrwGroups))
    print("adjNMI", adjustNMI(n, communityList, rnbrwGroups))

    # CLASSIFY COMMUNITIES by Cycle
    print("\n CYCLE")
    start = time()
    CNBRW(G, len(G.nodes))
    cycleGroups = nx.algorithms.community.louvain_communities(G, 'cycle', seed=100)
    print("CNBRW time n", time() - start)
    # print("Modularity", nx.algorithms.community.modularity(G, cycleGroups))
    # print("NMI", NMI(n, communityList, cycleGroups))
    print("adjNMI", adjustNMI(n, communityList, cycleGroups))

    # CLASSIFY COMMUNITIES by Weighted Cycle
    print("\n WEIGHTED CYCLE")
    start = time()
    weightedCNBRW(G, len(G.nodes))
    weightedCycleGroups = nx.algorithms.community.louvain_communities(G, 'weightedCycle', seed=100)
    print("CNBRW time n", time() - start)
    #print("Modularity", nx.algorithms.community.modularity(G, weightedCycleGroups))
    #print("NMI", NMI(n, communityList, weightedCycleGroups))
    print("adjNMI", adjustNMI(n, communityList, weightedCycleGroups))


def mainCycleStudy(n):
    G = LFRBenchmark(n)
    cycleData = cycleStudy(G, 1000)
    print(cycleData)


def createGraphs():
    for i in [1000, 5000, 10000]:
        n = i
        G = LFRBenchmark(n, average_degree=2 * log(n))
        string = "2ln_" + str(n)
        createGraphFiles.writeGraph(G, "graphs", string)
        print(string)
        communities = identifyLFRCommunities(G)
        createGraphFiles.writeCommunity(communities, "communities", string)

        # G = createGraphFiles.read("graphs/karate_club")
        # nx.draw(G, pos=nx.circular_layout(G))
        # plt.show()


def createGraphPackage(c=1):
    nodeList = [500, 1000, 5000, 10000]
    muList = [.1, .2, .3]
    for n in nodeList:
        for mu in muList:
            start = time()
            while time() - start < 30:
                G = LFRBenchmark(n, average_degree=c, mu=mu)
                if G is None:
                    continue
                string = str(c) + "_" + str(n) + "_" + str(mu)[-1]
                createGraphFiles.writeAll(G, string)
                print(string)
                break
            print(n, mu)


mainRetraceStudy(400)
# createGraphPackage(c=7)
