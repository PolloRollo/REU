"""
Create files storing networkx graphs
"""


from networkxTest import *
import os
import json
from time import time
from math import log
from networkxTest import identifyLFRCommunities


def writeGraph(G, folder, file):
    """Writes Graph data to a graph file"""
    string = folder + "/" + file + ".txt"
    nx.write_weighted_edgelist(G, string)
    return True


def writeCommunity(group, folder, file):
    """Writes community data to a community file"""
    string = folder + "/" + file + ".txt"
    f = open(string, "w")
    data = [list(g) for g in group]
    f.write(json.dumps(data))
    return True


def writeAll(G, file):
    """Writes all Graph data to separate graph and community files"""
    writeGraph(G, folder="graphs", file=file)
    group = identifyLFRCommunities(G)
    writeCommunity(group, folder="communities", file=file)
    return True


def readGraph(file):
    """ Return networkx graph G from stored file """
    if not os.access(file, 0):
        print("Error: Failed to access file")
    return nx.read_weighted_edgelist(file, nodetype=int)


def readCommunity(file):
    """Returns community data from stored file"""
    if not os.access(file, 0):
        print("Error: Failed to access file")
    f = open(file)
    data = json.load(f)
    newData = [set([int(i) for i in group]) for group in data]
    return newData


def readAll(file):
    """Automatically loads Graph and Community data from stored files"""
    G = readGraph("graphs/"+file)
    groups = readCommunity("communities/"+file)
    for group in groups:
        community = group
        for node in group:
            G.add_node(int(node), community=community)
    return G





