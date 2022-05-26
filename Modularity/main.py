"""



"""

from networkxTest import *
from graphData import *


def main(n):
    # Create community graph
    G = communityBuilder(n, 5, .7, .2)

    RNBRW(G, len(G.edges))
    CNBRW(G, floor(n * log(n)))
    print("rnbrw", nx.algorithms.community.louvain_communities(G, 'rnbrw_weight', 1))
    print("cycle", nx.algorithms.community.louvain_communities(G, 'cycle_rnbrw', 1))

    # rnbrw = [edge for edge in G.edges() if G[edge[0]][edge[1]]['rnbrw_weight'] > 0]
    # rnbrw = [edge for edge in G.edges() if G[edge[0]][edge[1]]['cycle_rnbrw'] > 1]
    # nx.draw(G, pos=nx.circular_layout(G))
    # nx.draw_networkx_nodes(G, pos=nx.circular_layout(G))
    # nx.draw_networkx_edges(G, pos=nx.circular_layout(G), edgelist=rnbrw, edge_color='r')
    # plt.show()


main(50)


