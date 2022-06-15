import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from scipy import stats


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
    a.set_title(str(x) + " 500 nodes with 10m iterations")
    a.set_xlabel("Estimated Retracing Probability")
    a.set_ylabel("Count")
    a.legend(["In Community Edge", "Across Community Edge"])
    # a.set(title='Retracing Probability')
    plt.show()


#compareAlgorithms("csvEdges/7_1000_3_100m.csv")
# compareWeights("csvEdges/7_1000_3_100m.csv")
# wassersteinDistance()
def createAllDirected(file):
    directedCompareAlgorithms(file)
    directedCompareWeights(file, "directed_rnbrw")
    directedCompareWeights(file, "directed_retrace")
    directedCompareWeights(file, "zigzag")
    directedCompareWeights(file, "zigzag_cycle")
    directedCompareWeights(file, "weighted_zigzag")


def allWasserstein():
    files = createFileList()
    iterations = ['1m', '10m', '100m']
    methods = ['directed_rnbrw', 'directed_retrace', 'zigzag', 'zigzag_cycle', 'weighted_zigzag']
    for i in iterations:
        for file in files:
            string = "csvEdgesDirected/" + file + "_" + i + ".csv"
            for method in methods:
                wasserstein(string, method)
            print()


def createFileList():
    nodes = ['500', '1000', '5000', '10000']
    mu = ['1', '2', '3', '4']
    degrees = ['1ln', '2ln', '3ln']
    fileList = []
    for node in nodes:
        for m in mu:
            for degree in degrees:
                string = degree + "_" + node + "_" + m
                fileList.append(string)
    return fileList


# compareAlgorithms("csvEdges/7_1000_3_100m.csv")
# createAll("csvEdges/7_1000_3_10m.csv")
# createAllDirected("csvEdgesDirected/1ln_500_3_10m.csv")
# print(createFileList())
allWasserstein()

