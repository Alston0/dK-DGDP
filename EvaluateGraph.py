import networkx as nx
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
def evaluate(G):

    # plt.tight_layout()
    # pos = nx.spring_layout(G, iterations=15, seed=1721)
    # fig, ax = plt.subplots(figsize=(15, 9))
    # plot_options = {"node_size": 10, "with_labels": True, "width": 0.15}
    # ax.axis("off")
    # nx.draw(G, pos=pos, ax=ax, **plot_options)
    # plt.show()
    # p = dict(nx.shortest_path_length(G))
    # print("最短路径长度shortest:",p)
    # lst = list(G.subgraph(c) for c in nx.weakly_connected_components(G))  # 提取图中所有连通子图，返回一个列表，默认按照结点数量由大到小排序
    # All_shortest = {}
    # for i in range(0, len(lst)):
    #     subgraph = lst[i]
    #     # -------------计算平均最短路径长度------------------
    #     shortest = nx.average_shortest_path_length(subgraph)
    #     All_shortest.setdefault(shortest,0)
    #     All_shortest[shortest]+=1
    #     #triangles = nx.triangles(subgraph)
    #     #print(np.average(list(triangles.values())))
    # print("平均最短路径长度shortest:", All_shortest)
    # plt.hist(All_shortest, bins=len(All_shortest),edgecolor="r")
    # plt.xlabel("shortest path length", size=20)  # Degree
    # plt.ylabel("Frequency", size=20)  # Frequency
    # plt.title('平均最短路径长度分布')
    # plt.show()

    # ----------------计算聚类系数cc和三角数量 ------------------------
    #triangles=nx.triangles(G)
    # cc = nx.average_clustering(G)
    # print("聚类系数(clustering coefficient):", cc)
    #print("三角形数量(triangles):", triangles)

    # ----------------计算degree centrality measures ------------------------
    # in_centrality = nx.in_degree_centrality(G)
    # out_centrality = nx.out_degree_centrality(G)
    # 
    # 
    # print("平均入度中心性(indegree centrality): ", np.average(list(in_centrality.values())))
    # print("平均出度中心性(outdegree centrality): ", np.average(list(out_centrality.values())))
    # 
    # # -------------计算Betweenness centrality measures------------------
    betweenness_centrality = nx.betweenness_centrality(G)
    print(betweenness_centrality)
    # print("平均中介中心度(Betweenness centrality): ", np.average(list(betweenness_centrality.values())))

    # -------------计算Average neighbor degree------------------
    # average_neighbor_outdegree = nx.average_neighbor_degree(G,target='out')
    # average_neighbor_indegree = nx.average_neighbor_degree(G,target='in')
    # print("平均邻居出度(Average neighbor outdegree): ", average_neighbor_outdegree)
    # print("平均邻居入度(Average neighbor indegree): ", average_neighbor_indegree)

    # # -------------计算Triad Census------------------
    # triad_Census = nx.triadic_census(G)
    # print("三角形计数(triad_Census): ", triad_Census)


g = nx.read_edgelist("D:\\pythonversion\\data\\p2p-Gnutella08.txt", create_using=nx.DiGraph)
# g = nx.read_edgelist("F:\\data\\twitter_combined.txt" ,create_using=nx.DiGraph)
evaluate(g)
