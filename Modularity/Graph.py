"""



"""
import Node
import networkx


class Graph:
    def __init__(self):
        self.nodes = []

    def adjacencyToGraph(self, adjacency):
        for i in range(len(adjacency)):
            self.nodes.append(Node.Node(i))

        for i in range(len(adjacency)):
            for j in range(len(adjacency)):
                if adjacency[i][j] > 1:
                    self.nodes[i].addEdge(self.nodes[j])

    def dictionaryToGraph(self, dictionary):
        nodes = []
        for i, _ in dictionary.items():
            self.nodes.append(Node.Node(i))
            nodes.append(i)

        for i in nodes:
            for j in range(dictionary[i]):
                self.nodes[i].addEdge(self.nodes[j])


