"""
Create files storing networkx graphs
"""


from networkxTest import *
import os
import json


def createLfrFile(n, average_degree, mu, file=None):
    G = LFRBenchmark(n=n, average_degree=average_degree, mu=mu)
    string = file
    if file is None:
        string = "LFR_" + str(n) + "_" + str(average_degree) + "_" + str(mu*10)
    nx.write_adjlist(G, "string")
    return True


def writeGraph(G, folder, file):
    string = folder + "/" + file
    nx.write_adjlist(G, string)
    return True


def writeCommunity(group, folder, file):
    string = folder + "/" + file
    f = open(string, "w")
    data = [list(g) for g in group]
    f.write(json.dumps(data))
    return True


def readGraph(file):
    """ Return networkx graph G from stored file """
    if not os.access(file, 0):
        print("Error: Failed to access file")
    return nx.read_adjlist(file)


def readCommunity(file):
    """Comment here"""
    if not os.access(file, 0):
        print("Error: Failed to access file")
    f = open(file)
    data = json.load(f)
    newData = [set([int(i) for i in group]) for group in data]
    return newData

