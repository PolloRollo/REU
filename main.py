"""



"""

from networkxTest import *
# from graphData import *
from time import time
from math import log, sqrt, floor
import createGraphFiles
# import directedlouvain


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
    # G = communityBuilder(n, floor(sqrt(n)), .45, .03)
    # G.remove_edges_from(nx.selfloop_edges(G))
    # IDENTIFY PRE-BUILT COMMUNITIES
    G = createGraphFiles.readAll("7_1000_3.txt")
    communityList = identifyLFRCommunities(G)

    # CLASSIFY COMMUNITIES unweighted
    print("\n UNWEIGHTED")
    start = time()
    unweightedGroups = nx.algorithms.community.louvain_communities(G, seed=100)
    print("Control time", time() - start)
    print("Modularity", nx.algorithms.community.modularity(G, unweightedGroups))
    # print("NMI", NMI(n, communityList, unweightedGroups))
    print("adjNMI", adjustNMI(n, communityList, unweightedGroups))

    # CLASSIFY COMMUNITIES by RNBRW
    print("\n RNBRW")
    start = time()
    RNBRW(G, t=10)
    rnbrwGroups = nx.algorithms.community.louvain_communities(G, 'rnbrw', seed=100)
    print("RNBRW time m", time() - start)
    print("Modularity", nx.algorithms.community.modularity(G, rnbrwGroups))
    #print("NMI", NMI(n, communityList, rnbrwGroups))
    print("adjNMI", adjustNMI(n, communityList, rnbrwGroups))

    # CLASSIFY COMMUNITIES by Cycle
    print("\n CYCLE")
    start = time()
    CNBRW(G, t=10)
    cycleGroups = nx.algorithms.community.louvain_communities(G, 'cycle', seed=100)
    print("CNBRW time n", time() - start)
    print("Modularity", nx.algorithms.community.modularity(G, cycleGroups))
    # print("NMI", NMI(n, communityList, cycleGroups))
    print("adjNMI", adjustNMI(n, communityList, cycleGroups))

    # CLASSIFY COMMUNITIES by Weighted Cycle
    print("\n WEIGHTED CYCLE")
    start = time()
    weightedCNBRW(G, t=10)
    weightedCycleGroups = nx.algorithms.community.louvain_communities(G, 'weightedCycle', seed=100)
    print("CNBRW time n", time() - start)
    print("Modularity", nx.algorithms.community.modularity(G, weightedCycleGroups))
    #print("NMI", NMI(n, communityList, weightedCycleGroups))
    print("adjNMI", adjustNMI(n, communityList, weightedCycleGroups))


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


def testCSVGraph(file, t=10):
    G = createGraphFiles.readAll(file)
    GraphToCSV(G, file, t=100)


def digraphTest(file, extra='', t=10):
    G = createGraphFiles.readDiAll(file, extra=extra)
    communities = identifyLFRCommunities(G)
    print(file, "Community Count", len(communities))
    # digraphLabeling(G)
    directedGraphToCSV(G, file, extra=extra, t=t)


def createAllEdgeCSVs(directory="digraphs/networks/", extra="", t=1):
    for file in os.listdir(directory):
        file = file[:-4]
        digraphTest(file, extra=extra, t=t)


def reciprocalEdge(file):
    G = createGraphFiles.readDiAll(file)
    count = 0
    for head, tail in G.edges:
        if tail in G[head] and head in G[tail]:
            count += 1
    print(count//2)
    print(len(G.edges))
    return count//2


def writeAllWeights(file, folder="weightedDigraphs",extra='', t=1):
    G = createGraphFiles.readDiAll(file, extra=extra)
    # digraphLabeling(G)
    methods = ['directed_rnbrw', 'backtrack', 'zigzag', 'zigzag_cycle', 'weighted_zigzag', 'directed_cycle']
    print(file)
    # methods = ['directed_rnbrw', 'backtrack', 'directed_cycle']
    DRNBRW(G, t)
    backtrackDRW(G, t)
    ZRNBRW(G, t)
    ZCNBRW(G, t)
    weightedZCNBRW(G, t)
    directedCycle(G, t)
    for method in methods:
        createGraphFiles.writeGraphWeights(G, folder=folder, file=file, method=method, extra=extra, t=t)


def createAllWeightFiles(directory="digraphs/networks/", extra='', t=1):
    for file in os.listdir(directory):
        file = file[:-4]
        writeAllWeights(file, extra=extra, t=t)


def randomWalkPath(G):
    tail = random.choice(list(G.nodes))
    path = [tail]
    head = random.choice(list(G.successors(tail)))
    edges = [[tail, head]]
    while head not in path:
        neighbors = list(G.successors(head))
        if tail in neighbors:
            neighbors.remove(tail)
        if len(neighbors) == 0:
            return False, path, edges
        path.append(head)
        tail, head = head, random.choice(neighbors)
        edges.append([tail, head])
    print(head)
    return True, path, edges


def subdigraphTest(file):
    G = createGraphFiles.readDiAll(file)
    # plt.figure(15, figsize=(60, 60))
    complete, path, edges = randomWalkPath(G)
    print(path)
    H = G.subgraph(path)
    nx.set_edge_attributes(G, values=0, name='path')
    # for edge1, edge2 in edges:
        # G[edge1][edge2]['path'] = 10
    nx.set_edge_attributes(H, values=1, name='path')
    edges, weights = zip(*nx.get_edge_attributes(H, 'path').items())
    # print(edges)
    color_map = [(i in path) for i in range(len(G.nodes))]
    pos = nx.kamada_kawai_layout(G)
    # nx.draw(G, pos, node_size=20, node_color=color_map, arrowsize=5)
    nx.draw_networkx_edges(G, pos, edgelist=edges, width=1.0, edge_color='y', style='solid')
    plt.show()
    # subgraphRNBRW(G)

"""
def thesaurus(file='EATnew/EATgraph.txt', t=1):
    G = nx.read_weighted_edgelist(file, create_using=nx.DiGraph, nodetype=int)
    methods = ['directed_rnbrw', 'backtrack', 'zigzag', 'zigzag_cycle', 'weighted_zigzag', 'directed_cycle']
    # print(file)
    # methods = ['directed_rnbrw', 'backtrack', 'directed_cycle']
    DRNBRW(G, t)
    backtrackDRW(G, t)
    ZRNBRW(G, t)
    ZCNBRW(G, t)
    weightedZCNBRW(G, t)
    directedCycle(G, t)
    for method in methods:
        createGraphFiles.writeGraphWeights(G, folder='weightedDigraphs/thesaurusWeighted', file='EATgraph', method=method, t=t)
    with open("EATnew/EATcsv_" + str(t) + "m.csv", 'w') as csvfile:
        fieldnames = ['in_comm', 'directed_rnbrw', 'backtrack', 'directed_cycle', 'zigzag', 'zigzag_cycle', 'weighted_zigzag']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for head, tail in G.edges:
            writer.writerow({'directed_rnbrw': G[head][tail]['directed_rnbrw'],
                             'backtrack': G[head][tail]['backtrack'],
                             'directed_cycle': G[head][tail]['directed_cycle'],
                             'zigzag': G[head][tail]['zigzag'],
                             'zigzag_cycle': G[head][tail]['zigzag_cycle'],
                             'weighted_zigzag': G[head][tail]['weighted_zigzag']})

edges, weights = zip(*nx.get_edge_attributes(G, weight).items())
nx.draw(G, edgelist=edges, edge_color=weights)
"""


def createRealGraphs(directory="realGraphs/Original", extra='', t=1):
    for file in os.listdir(directory):
        if file == '.DS_Store':
            continue
        file = directory + "/" + file
        writeAllReal(file, t=t)


def writeAllReal(file, folder="realGraphs/Weighted", t=1):
    G = createGraphFiles.readDiReal(file)
    # digraphLabeling(G)
    methods = ['directed_rnbrw', 'backtrack', 'zigzag', 'zigzag_cycle', 'weighted_zigzag', 'directed_cycle']
    file = file.split('/')[-1]
    print(file)
    # methods = ['directed_rnbrw', 'backtrack', 'directed_cycle']
    DRNBRW(G, t)
    backtrackDRW(G, t)
    ZRNBRW(G, t)
    ZCNBRW(G, t)
    weightedZCNBRW(G, t)
    directedCycle(G, t)
    with open(folder + '/' + file.split('.')[0] + str(t) + "m.csv", 'w') as csvfile:
        fieldnames = ['directed_rnbrw', 'backtrack', 'directed_cycle', 'zigzag', 'zigzag_cycle', 'weighted_zigzag']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for head, tail in G.edges:
            writer.writerow({'directed_rnbrw': G[head][tail]['directed_rnbrw'],
                             'backtrack': G[head][tail]['backtrack'],
                             'directed_cycle': G[head][tail]['directed_cycle'],
                             'zigzag': G[head][tail]['zigzag'],
                             'zigzag_cycle': G[head][tail]['zigzag_cycle'],
                             'weighted_zigzag': G[head][tail]['weighted_zigzag']})


def subgraphTest(file):
    G = createGraphFiles.readAll(file)
    nx.draw(G)
    plt.show()


def reciprocityTest(directory="realGraphs/Original"):
    for file in os.listdir(directory):
        if file == '.DS_Store':
            continue
        file = directory + "/" + file
        G = createGraphFiles.readDiReal(file)
        print(file)
        print("N:", len(G.nodes))
        print("E:", len(G.edges))
        print("r:", reciprocalRatio(G))
        print("rho:", reciprocityIndex(G))


def final(graphDirectory="realGraphs/Original", commDirectory="realGraphs/finalCommunities"):
    dict = {'moreno_highschool.txt': 'highschool',
            'moreno_health.txt': 'health',
            'moreno_oz.txt': 'college'}
    for file in os.listdir(graphDirectory):
        if file == '.DS_Store':
            continue
        if file in dict:
            print(file)
            graphPath = graphDirectory + '/' + file
            commPath = commDirectory + '/' + dict[file]
            G = createGraphFiles.readDiReal(graphPath)
            for commFile in os.listdir(commPath):
                if commFile == '.DS_Store':
                    continue
                commPathFile = commPath + '/' + commFile
                communities = createGraphFiles.readDiCommunities(commPathFile)
                fileList = commFile.split('_')
                string = ''
                types = ''
                if len(fileList) == 2:
                    # Only Louvain
                    string = '---'
                    types = '---'
                else:
                    string = fileList[-2][4:]
                    if len(fileList) == 5:
                        types = 'r'
                        continue
                    elif len(fileList) == 6:
                        types = 'c'
                print(finalTest(G, communities, string) + '\\\\')
                # print(string, communityCounter(communities))


def finalCSV(graphDirectory="realGraphs/Original", commDirectory="realGraphs/finalCommInc", folder='realGraphs/CSV'):
    dict = {'moreno_highschool.txt': 'highschool',
            'moreno_health.txt': 'health',
            'moreno_oz.txt': 'college'}
    for file in os.listdir(graphDirectory):
        if file == '.DS_Store':
            continue
        if file in dict:
            print(file)
            with open(folder + '/' + file + ".csv", 'w') as csvfile:
                fieldnames = ['network', 'initial', 'singleton', 'commTotal', 'commMax', 'commAve', 'r', 'rho']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                graphPath = graphDirectory + '/' + file
                commPath = commDirectory + '/' + dict[file]
                G = createGraphFiles.readDiReal(graphPath)
                for commFile in os.listdir(commPath):
                    if commFile == '.DS_Store':
                        continue
                    commPathFile = commPath + '/' + commFile
                    communities = createGraphFiles.readDiCommunities(commPathFile)
                    fileList = commFile.split('_')
                    string = ''
                    if len(fileList) == 2:
                        # Only Louvain
                        string = '---'
                        continue
                    else:
                        string = fileList[-2][4:]
                    data = finalTest(G, communities, string, latex=False)
                    writer.writerow({'network': dict[file],
                                     'initial': data[0],
                                     'singleton': data[1],
                                     'commTotal': data[2],
                                     'commMax': data[3],
                                     'commAve': data[4],
                                     'r': data[5],
                                     'rho': data[6]})



final()
# reciprocityTest()
# subdigraphTest("1ln_1000_1")
# reciprocalEdge("1ln_10000_3")
# mainRetraceStudy(1000)
# createGraphPackage(c=7)
# testCSVGraph()
# testCSVGraph("7_1000_3", 100)
# digraphTest("10ln_500_3", t=1)
# createAllEdgeCSVs("digraphs/small_tau/networks/", extra='small_tau', t=1)
# createAllWeightFiles(directory="digraphs/larger_community_range/networks/", extra='larger_community_range', t=10)
# createRealGraphs(t=10)

