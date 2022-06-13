import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from scipy import stats


def compareAlgorithms(file):
    df = pd.read_csv(file)
    a = sns.pairplot(df, hue="in_comm", diag_kind="hist")
    a.fig.suptitle("Algorithm Comparison: 100m iterations", fontsize=18)
    handles = a._legend_data.values()
    labels = ["Across-Community Edge", "In-Community Edge"]
    sns.move_legend(a, (2, 2))
    a.set(xlim=(0, .002), ylim=(0, .002))
    a.fig.legend(handles=handles, labels=labels, loc="upper left", ncol=1)
    plt.show()


def compareWeights(file):
    df = pd.read_csv(file)
    a = sns.histplot(df, x='cycle', hue='in_comm', multiple='layer')
    a.set_title("Cycle with 100m iterations")
    a.set_xlabel("Estimated Retracing Probability")
    a.set_ylabel("Count")
    a.legend(["In Community Edge", "Across Community Edge"])
    # a.set(title='Retracing Probability')
    plt.show()


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
    a = sns.histplot(df, x=x, hue='in_comm', multiple='layer')
    a.set_title(str(x) + "10000 nodes with 10m iterations")
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
    directedCompareWeights(file, "zigzag")
    directedCompareWeights(file, "zigzag_cycle")
    directedCompareWeights(file, "weighted_zigzag")



# compareAlgorithms("csvEdges/7_1000_3_100m.csv")
createAllDirected("csvEdgesDirected/1ln_10000_3_10m.csv")
