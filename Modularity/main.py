"""



"""

from networkxTest import *
from graphData import *
from time import time


def main(n, group_count=25, draw=False):
    # Create community graph
    print(n)
    start = time()
    #G = communityBuilder(n, group_count, p_in=.7, p_out=.2)
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
    RNBRW(G, len(G.edges))
    rnbrw = nx.algorithms.community.louvain_communities(G, 'rnbrw', 1)
    # print("rnbrw", rnbrw)
    print("number of groups", len(rnbrw))
    print("RNBRW time m", time() - start)

    start = time()
    CNBRW(G, n)
    cycle = nx.algorithms.community.louvain_communities(G, 'cycle', 1)
    # print("cycle", cycle)
    print("number of groups", len(cycle))
    print("Cycle time n", time() - start)

    start = time()
    CNBRW(G, 2 * n)
    cycle = nx.algorithms.community.louvain_communities(G, 'cycle', 1)
    # print("cycle", cycle)
    print("number of groups", len(cycle))
    print("Cycle time 2n", time() - start)

    start = time()
    CNBRW(G, floor(n * log(n)))
    cycle = nx.algorithms.community.louvain_communities(G, 'cycle', 1)
    # print("cycle", cycle)
    print("number of groups", len(cycle))
    print("Cycle time nlogn", time() - start)

    start = time()
    CNBRW(G, len(G.edges))
    cycle = nx.algorithms.community.louvain_communities(G, 'cycle', 1)
    # print("cycle", cycle)
    print("number of groups", len(cycle))
    print("Cycle time m", time() - start)

    if draw:
        rnbrw = [edge for edge in G.edges() if G[edge[0]][edge[1]]['rnbrw'] > 0]
        # rnbrw = [edge for edge in G.edges() if G[edge[0]][edge[1]]['cycle'] > 1]
        # nx.draw(G, pos=nx.circular_layout(G))
        nx.draw_networkx_nodes(G, pos=nx.circular_layout(G))
        nx.draw_networkx_edges(G, pos=nx.circular_layout(G), edgelist=rnbrw, edge_color='r')
        plt.show()


for i in range(10):
    main(100 * 2**i)


