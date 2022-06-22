""""
David Rollo's code testing graph creation methods
"""

import random
import networkx as nx
import createGraphFiles
from networkxTest import *


def clusterTest(file, t=10):
    G = createGraphFiles.readAll(file)
    communityList = identifyLFRCommunities(G)
    weightedCNBRW(G, t)
    edgeIndexList = [i for i in range(len(G.edges))]
    edgeWeightList = []
    edgePositionList = []
    for tail, head in G.edges:
        edgeWeightList.append(G[tail][head]['weightedCycle'])
        edgePositionList.append([tail, head])








