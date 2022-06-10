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
        print("Error: Failed to access graph file")
    return nx.read_weighted_edgelist(file, nodetype=int)


def readCommunity(file):
    """Returns community data from stored file"""
    if not os.access(file, 0):
        print("Error: Failed to access community file")
    f = open(file)
    data = json.load(f)
    newData = [set([int(i) for i in group]) for group in data]
    return newData


def readAll(file):
    """Automatically loads Graph and Community data from stored files"""
    if str(file[-4]) != ".txt":
        file += ".txt"
    G = readGraph("graphs/"+file)
    groups = readCommunity("communities/"+file)
    for group in groups:
        community = group
        for node in group:
            G.add_node(int(node), community=community)
    return G


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
    if str(file[-4]) != ".dat":
        file += ".dat"
    G = readDiGraph("digraphs/networks/"+file)
    groups = readDiCommunities("digraphs/communities/"+file)

    for group in groups:
        community = group
        for node in group:
            G.add_node(int(node), community=community)
    return G
