import pandas as pd
# import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from scipy import stats
import os
import numpy as np


def compareAlgorithms(file):
    df = pd.read_csv(file)
    a = sns.pairplot(df, hue="in_comm", diag_kind="hist")
    a.fig.suptitle("Algorithm Comparison: 10m iterations", fontsize=18)
    handles = a._legend_data.values()
    labels = ["Across-Community Edge", "In-Community Edge"]
    sns.move_legend(a, (2, 2))
    a.set(xlim=(0, .002), ylim=(0, .002))
    a.fig.legend(handles=handles, labels=labels, loc="upper left", ncol=1)
    plt.show()


def compareWeights(file, x):
    df = pd.read_csv(file)
    a = sns.histplot(df, x=x, hue='in_comm', multiple='dodge')
    a.set_title(x + " 1000 nodes with 10m iterations")
    a.set_xlabel("Estimated Retracing Probability")
    a.set_ylabel("Count")
    a.legend(["In Community Edge", "Across Community Edge"])
    # a.set(title='Retracing Probability')
    plt.show()


def createAll(file):
    compareAlgorithms(file)
    compareWeights(file, "rnbrw")
    compareWeights(file, "cycle")
    compareWeights(file, "weightedCycle")
    compareWeights(file, "hybrid")


def wassersteinDistance():
    df1 = pd.read_csv("csvEdges/7_1000_3_1m.csv")
    df10 = pd.read_csv("csvEdges/7_1000_3_10m.csv")
    df100 = pd.read_csv("csvEdges/7_1000_3_100m.csv")

    df1_in = df1[df1['in_comm'] == True]
    df1_out = df1[df1['in_comm'] == False]

    df10_in = df10[df10['in_comm'] == True]
    df10_out = df10[df10['in_comm'] == False]

    df100_in = df100[df100['in_comm'] == True]
    df100_out = df100[df100['in_comm'] == False]

    print("1, 10, 100: Rnbrw")
    print(stats.wasserstein_distance(df1_in['rnbrw'], df1_out['rnbrw']))
    print(stats.wasserstein_distance(df10_in['rnbrw'], df10_out['rnbrw']))
    print(stats.wasserstein_distance(df100_in['rnbrw'], df100_out['rnbrw']))

    print("1, 10, 100: Cycle")
    print(stats.wasserstein_distance(df1_in['cycle'], df1_out['cycle']))
    print(stats.wasserstein_distance(df10_in['cycle'], df10_out['cycle']))
    print(stats.wasserstein_distance(df100_in['cycle'], df100_out['cycle']))

    print("1, 10, 100: Weighted Cycle")
    print(stats.wasserstein_distance(df1_in['weightedCycle'], df1_out['weightedCycle']))
    print(stats.wasserstein_distance(df10_in['weightedCycle'], df10_out['weightedCycle']))
    print(stats.wasserstein_distance(df100_in['weightedCycle'], df100_out['weightedCycle']))


def wasserstein(file, method):
    df = pd.read_csv(file)
    df_in = df[df['in_comm'] == True]
    df_out = df[df['in_comm'] == False]
    print(file, method, stats.wasserstein_distance(df_in[method], df_out[method]))


def directedCompareAlgorithms(file):
    df = pd.read_csv(file)
    a = sns.pairplot(df, hue="in_comm", diag_kind="hist")
    a.fig.suptitle("Algorithm Comparison: 10m iterations", fontsize=18)
    handles = a._legend_data.values()
    labels = ["Across-Community Edge", "In-Community Edge"]
    sns.move_legend(a, (2, 2))
    # a.set(xlim=(0, .002), ylim=(0, .002))
    a.fig.legend(handles=handles, labels=labels, loc='upper left', ncol=1)
    plt.show()


def directedCompareWeights(file, x):
    df = pd.read_csv(file)
    # print(df.head())
    # string = x + "_log"
    # df[string] = df[x].apply(np.log)
    a = sns.histplot(df, x=x, hue='in_comm', multiple='dodge',  bins=50)
    a.set_title(str(x) + " .8 mu 1000 nodes with 10m iterations")
    a.set_xlabel("Estimated Retracing Probability")
    a.set_ylabel("Count")
    a.legend(["In Community Edge", "Across Community Edge"])
    # a.set(title='Retracing Probability')
    plt.show()


def realDirectedCompareWeights(file, x):
    df = pd.read_csv(file)
    a = None
    if file == "realGraphs/Weighted/EATgraph1m.csv" and x == 'zigzag_cycle':
        a = sns.histplot(df, x=x, binwidth=2, binrange=[0, 100])
        (n, bins, patches) = plt.hist(df[x], bins=100, range=(0, 100), label='hst')
        print(n)
    else:
        a = sns.histplot(df, x=x, binwidth=1, binrange=[0, 30])
        (n, bins, patches) = plt.hist(df[x], bins=30, range=(0, 30), label='hst')
        print(n)
        # print(plt.hist(x, bins=30, range=(0, 30), label='hst'))
    a.set_title(str(x) + ' ' + str(file))
    a.set_xlabel("Estimated Retracing Probability")
    a.set_ylabel("Count")
    # a.legend(["In Community Edge", "Across Community Edge"])
    # a.set(title='Retracing Probability')
    plt.show()


def realCreateAllDirected(file):
    realDirectedCompareWeights(file, "directed_rnbrw")
    # realDirectedCompareWeights(file, "directed_cycle")
    # realDirectedCompareWeights(file, 'backtrack')
    realDirectedCompareWeights(file, "zigzag")
    # realDirectedCompareWeights(file, "zigzag_cycle")
    realDirectedCompareWeights(file, "weighted_zigzag")


def createAllDirected(file):
    directedCompareAlgorithms(file)
    directedCompareWeights(file, "directed_rnbrw")
    directedCompareWeights(file, "directed_cycle")
    directedCompareWeights(file, 'backtrack')
    directedCompareWeights(file, "zigzag")
    directedCompareWeights(file, "zigzag_cycle")
    directedCompareWeights(file, "weighted_zigzag")


def allWasserstein():
    files = os.listdir("digraphs/networks/")
    iterations = ['1m', '10m', '100m']
    methods = ['directed_rnbrw', 'backtrack', 'zigzag', 'zigzag_cycle', 'weighted_zigzag']
    for i in iterations:
        for file in files:
            string = "csvEdgesDirected/" + file + "_" + i + ".csv"
            for method in methods:
                wasserstein(string, method)
            print()


def edgeStatistics(file):
    df = pd.read_csv(file)
    nameList = file.replace('_', ' ').replace('/', ' ').replace('.', ' ').split()
    methodDf = pd.DataFrame({'file': [file.split(sep="/")[1]],
                             'nodes': [int(nameList[2])],
                             'edges': [int(nameList[1][:-2])],
                             'mixing': [int(nameList[3][0])],
                             'iter': [int(nameList[-2][:-1])]})
    methods = ['directed_rnbrw', 'backtrack', 'zigzag', 'zigzag_cycle', 'weighted_zigzag']
    for method in methods:
        methodDf[str('in_comm_' + method + '_mean')] = df[df['in_comm'] == True][method].mean()
        methodDf[str('in_comm_' + method + '_std')] = df[df['in_comm'] == True][method].median()
        methodDf[str('in_comm_' + method + '_median')] = df[df['in_comm'] == True][method].std()
        methodDf[str('in_comm_' + method + '_max')] = df[df['in_comm'] == True][method].max()
        methodDf[str('in_comm_' + method + '_min')] = df[df['in_comm'] == True][method].min()
        methodDf[str('out_comm_' + method + '_mean')] = df[df['in_comm'] == False][method].mean()
        methodDf[str('out_comm_' + method + '_std')] = df[df['in_comm'] == False][method].median()
        methodDf[str('out_comm_' + method + '_median')] = df[df['in_comm'] == False][method].std()
        methodDf[str('out_comm_' + method + '_max')] = df[df['in_comm'] == False][method].max()
        methodDf[str('out_comm_' + method + '_min')] = df[df['in_comm'] == False][method].min()
    return methodDf


def createMetaAnalysis(directory="csvEdgesDirected"):
    metaDF = None
    for filename in os.listdir(directory):
        if filename == '.DS_Store':
            continue
        f = os.path.join(directory, filename)
        print(f)
        methodDF = edgeStatistics(f)
        if metaDF is None:
            metaDF = methodDF
        else:
            metaDF = pd.concat([metaDF, methodDF])
    metaDF.to_csv("metaEdges.csv")


def performMetaAnalysis():
    df = pd.read_csv("metaEdges.csv")
    sns.color_palette("husl", 8)
    a = sns.scatterplot(data=df,
                        x=df['in_comm_zigzag_mean']/df['iter'],
                        y=df['out_comm_zigzag_mean']/df['iter'],
                        hue= np.log(df['nodes']),
                        palette='Spectral')
    a.set_title("Analysis of Zigzag Mean Weights Colored by Node Count")
    a.set_xlabel("Mean In-Community Weight")
    a.set_ylabel("Mean Across-Community Weight")
    a.legend(['0', '500', '1000', '5000', '10000'])
    # a.legend(['0', '.1', '.2', '.3', '.4', '.6', '.8'])

    plt.show()


def allRealGraphs(directory='realGraphs/Weighted'):
    for filename in os.listdir(directory):
        if filename == '.DS_Store':
            continue
        f = os.path.join(directory, filename)
        print(f)
        realCreateAllDirected(f)


# performMetaAnalysis()
allRealGraphs()
# realCreateAllDirected("EATnew/EATcsv_1m.csv")
# createMetaAnalysis()
# compareAlgorithms("csvEdges/7_1000_3_100m.csv")
# createAll("csvEdges/7_1000_3_10m.csv")
# createAllDirected("csvEdgesDirected/1ln_1000_8_10m.csv")
# createAllDirected("EATnew/EATcsv_1m.csv")
# print(createFileList())
# allWasserstein()

