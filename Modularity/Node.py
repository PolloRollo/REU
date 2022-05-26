"""
Each Node is an object in a graph. The edge list is a list of neighboring Nodes



"""


class Node:
    def __init__(self, id):
        self.id = id
        self.edges = set()
        self.outDegree = 0

    def addEdge(self, node):
        self.edges.add(node)
        self.outDegree += 1

    def removeEdge(self, node):
        if node in self.edges:
            self.edges -= node
            self.outDegree -= 1

