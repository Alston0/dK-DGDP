import matplotlib.pyplot as plt
import networkx as nx
from collections import Counter
import numpy as np
import heapq

# def node_degree_joint(G):
#     for i in range(len(list(G.edges()))):
#         degu=G.out_degree[list(G.edges())[i][0]]
#         degv=G.in_degree[list(G.edges())[i][1]]
#         yield degu,degv
#
#
# def getdegnode(degree, degreetype, in_degrees, out_degrees):
#     sample = []
#     if degreetype == 0:
#         for i in range(len(in_degrees)):
#             if in_degrees[i] == degree:
#                 sample.append(i)
#     if degreetype == 1:
#         for i in range(len(out_degrees)):
#             if out_degrees[i] == degree:
#                 sample.append(i)
#     return sample


def show_dis(histogram):
    x = range(len(histogram))
    y = [z / float(sum(histogram)) for z in histogram]
    # plt.xlabel('histogram', size=20)  # Degree    #222222222222222222222222222222222222
    # plt.ylabel("Frequency", size=20)  # Frequency
    # plt.xticks(fontproperties='Times New Roman', size=13)
    # plt.yticks(fontproperties='Times New Roman', size=13)
    # plt.loglog(x, y, color="blue", linewidth=1)
    # plt.show()    #222222222222222222222222222222222222

def variation(y,gap):
    li=list()
    for i in range(gap,len(y)-gap):
        li.append(np.var(y[i-gap:i])/np.var(y[i:i+gap]))
    return li

def indegree_histogram(G):
    """Returns a list of the frequency of each indegree value.

    Parameters
    ----------
    G : Networkx graph
       A graph

    Returns
    -------
    hist : list
       A list of frequencies of indegrees.
       The degree values are the index in the list.

    Notes
    -----
    Note: the bins are width one, hence len(list) can be large
    (Order(number_of_edges))
    """
    counts = Counter(d for n, d in G.in_degree())
    return [counts.get(i, 0) for i in range(max(counts) + 1)]

def outdegree_histogram(G):
    """Returns a list of the frequency of each outdegree value.

    Parameters
    ----------
    G : Networkx graph
       A graph

    Returns
    -------
    hist : list
       A list of frequencies of outdegrees.
       The degree values are the index in the list.

    Notes
    -----
    Note: the bins are width one, hence len(list) can be large
    (Order(number_of_edges))
    """
    counts = Counter(d for n, d in G.out_degree())
    return [counts.get(i, 0) for i in range(max(counts) + 1)]







# # 计算每条边的重要程度，这里想的是由出度大但入度小的点指向入度大但出度小的点的边为不重要的,反之则为重要
# for i in range(0,len(g.edges)):
#     source_node=int(edges[i][0])
#     destination_node=int(edges[i][1])
#     edgescore[i]=(nodedegseq[source_node][0]+nodedegseq[destination_node][1]-nodedegseq[source_node][1]-nodedegseq[destination_node][0])






# outdegree_histogram=outdegree_histogram(g)
# indegree_histogram=indegree_histogram(g)
# show_dis(outdegree_histogram)
# show_dis(indegree_histogram)




def split(g,n):
    nodedegseq = {}
    splitlist=[]
    for i in g.nodes:
        nodedegseq[int(i)] = (g.in_degree(i), g.out_degree(i))
    edgescore = {}
    edges = list(g.edges())
    # 计算每条边的重要程度，度大的节点重要
    for i in range(0, len(g.edges)):
        source_node = int(edges[i][0])
        destination_node = int(edges[i][1])
        edgescore[i] = (nodedegseq[source_node][0] + nodedegseq[destination_node][1] + nodedegseq[source_node][1] +
                        nodedegseq[destination_node][0])
        sorted_score = sorted(edgescore.items(), key=lambda edge: edge[1], reverse=True)
    x = range(len(sorted_score))
    y = [sorted_score[z][1] for z in x]
    # plt.xlabel('edge', size=20)  # Degree   #222222222222222222222222222222222222
    # plt.ylabel("edge_score", size=20)  # Frequency
    # plt.xticks(fontproperties='Times New Roman', size=13)
    # plt.yticks(fontproperties='Times New Roman', size=13)
    # plt.plot(x, y, color="blue", linewidth=1)
    # plt.show()    #222222222222222222222222222222222222
    # 求分组区间，输出离散程度最大的几个点
    var_score = variation(sorted_score, 3)
    x = range(0,len(var_score))
    y = [var_score[z] for z in x]
    # plt.xlabel('Edge_score', size=20)  # Degree     #222222222222222222222222222222222222
    # plt.ylabel("Variation", size=20)  # Frequency
    # plt.xticks(fontproperties='Times New Roman', size=13)
    # plt.yticks(fontproperties='Times New Roman', size=13)
    # plt.plot(x, y, color="blue", linewidth=1)
    # plt.show()    #222222222222222222222222222222222222
    #print(var_score)    #22222222222222222222222222222
    var_score_sort = sorted(var_score)
    # print(var_score_sort)  #22222222222222222222222222222
    # print(var_score_sort[len(var_score_sort)-1])
    # print(var_score_sort[len(var_score_sort) -2])
    # print(var_score_sort[len(var_score_sort) - 3])
    # print(var_score.index(var_score_sort[len(var_score_sort)-1]))
    # print(var_score.index(var_score_sort[len(var_score_sort)-2]))
    # print(var_score.index(var_score_sort[len(var_score_sort) - 3]))  #22222222222222222222222222222
    for i in range(0,n-1):
        splitlist[i]=var_score.index(var_score_sort[len(var_score_sort)-i-1])
    return splitlist

g = nx.read_edgelist("/media/HD0/dkdgdp/data/email-dnc.txt" , create_using=nx.DiGraph)
split(g,3)

