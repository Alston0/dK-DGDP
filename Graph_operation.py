from collections import Counter
import networkx as nx
import math
import numpy as np
import random
def indegree_histogram(G):
    """Returns a list of the frequency of each degree value.

    Parameters
    ----------
    G : Networkx graph
       A graph

    Returns
    -------
    hist : list
       A list of frequencies of degrees.
       The degree values are the index in the list.

    Notes
    -----
    Note: the bins are width one, hence len(list) can be large
    (Order(number_of_edges))
    """
    counts = Counter(d for n, d in G.in_degree())
    return [counts.get(i, 0) for i in range(max(counts) + 1)]

def outdegree_histogram(G):
    """Returns a list of the frequency of each degree value.

    Parameters
    ----------
    G : Networkx graph
       A graph

    Returns
    -------
    hist : list
       A list of frequencies of degrees.
       The degree values are the index in the list.

    Notes
    -----
    Note: the bins are width one, hence len(list) can be large
    (Order(number_of_edges))
    """
    counts = Counter(d for n, d in G.out_degree())
    return [counts.get(i, 0) for i in range(max(counts) + 1)]


def node_degree_joint(G):
    for i in range(len(list(G.edges()))):
        out_degu=G.out_degree[list(G.edges())[i][0]]
        in_degu=G.in_degree[list(G.edges())[i][0]]
        out_degv = G.out_degree[list(G.edges())[i][1]]
        in_degv = G.in_degree[list(G.edges())[i][1]]
        yield (out_degu,in_degu),(out_degv,in_degv)

def getQuintuple(g):
    JDM={}
    edgesdegree_list = list(node_degree_joint(g))
    for i in range(len(edgesdegree_list)):
        JDM[edgesdegree_list[i]]=JDM.get((edgesdegree_list[i]),0)+1
        # JDM.setdefault([g.in_degree[g.edges[i][0]],g.out_degree[i][0]],{})  #44444444444444444444444444444444444
        # JDM[[g.in_degree[i][0],g.out_degree[i][0]]].setdefault([g.in_degree[i][1],g.out_degree[i][1]],0)  #44444444444444444444444444444444444
    return JDM

def DK2(JDM):
    h_jointdegree_edgecount_in = {}
    h_jointdegree_edgecount_out = {}
    # for a given JDM, keep the list of all node ids.
    h_jointdegree_nodelist_in={}
    h_jointdegree_nodelist_out = {}
    # for a given JDM,get the list of the nodes with joint degrees
    h_jointdegree_nodelist={}

    # count the node number between the indegree list and the oudegree list,normally they should be the same
    nodecount={}
    # for a node with degree(outdeg,indeg),not satified sum(nkk[k])/k = number of nodes
    out_unsat={}
    in_unsat={}
    # print(JDM)
    # 获取JDM中出度节点的出度边数，入度节点的入度边数，并记录其match的入度和出度
    JDM_items = list(JDM.items())
    # print(JDM_items)
    for i in range(len(JDM_items)):
        h_jointdegree_edgecount_in [JDM_items[i][0][1]] = h_jointdegree_edgecount_in .get(JDM_items[i][0][1], 0) + JDM_items[i][1]
        h_jointdegree_edgecount_out[JDM_items[i][0][0]] = h_jointdegree_edgecount_out.get(JDM_items[i][0][0], 0) + JDM_items[i][1]

    for i in h_jointdegree_edgecount_in.keys():
        h_jointdegree_nodelist_in[i]=h_jointdegree_nodelist_in.get(i,0)+math.ceil(h_jointdegree_edgecount_in[i]/i[1])
        h_jointdegree_nodelist[i]= h_jointdegree_nodelist.get(i,h_jointdegree_nodelist_in[i])
        nodecount.setdefault(i,{})
        nodecount[i][1]=h_jointdegree_nodelist_in[i]

    for i in h_jointdegree_edgecount_out.keys():
        h_jointdegree_nodelist_out[i]=h_jointdegree_nodelist_out.get(i,0)+math.ceil(h_jointdegree_edgecount_out[i]/i[0])
        h_jointdegree_nodelist[i]=h_jointdegree_nodelist.get(i,h_jointdegree_nodelist_out[i])
        nodecount.setdefault(i, {})
        nodecount[i][0] = h_jointdegree_nodelist_out[i]
        if i[1]==0:
            h_jointdegree_nodelist[i]=h_jointdegree_nodelist_out[i]
    # print(nodecount)

    # 处理多边问题
    for i in range(len(JDM_items)):
        if h_jointdegree_nodelist_in[JDM_items[i][0][1]]*h_jointdegree_nodelist_out[JDM_items[i][0][0]]<JDM_items[i][1]:
            JDM[JDM_items[i][0]]=h_jointdegree_nodelist_in[JDM_items[i][0][1]]*h_jointdegree_nodelist_out[JDM_items[i][0][0]]


    # 考虑以下情况：某节点v(outdeg,indeg),由outdeg计算节点个数为x个，由indeg计算节点个数为y个，需要统一为大的那个 并且进一步处理
    for i in nodecount.keys():
        if len(nodecount[i])>1:
            # 如果入度计算的节点个数大于出度计算的节点个数，统一成入度节点个数，并添加入度
            if nodecount[i][1] > nodecount[i][0]:
                # 节点数*入度>入边
                if h_jointdegree_edgecount_in[i]<i[1]*nodecount[i][1]:
                    # in_unsat.setdefault(i, i[1]*nodecount[i][1]-h_jointdegree_edgecount_in[i])
                    in_unsat[i]=in_unsat.get(i,0)+i[1]*nodecount[i][1]-h_jointdegree_edgecount_in[i]
                if h_jointdegree_edgecount_out[i]<i[0]*nodecount[i][1]:
                    out_unsat[i] = out_unsat.get(i, 0) + i[0] * nodecount[i][1] - h_jointdegree_edgecount_out[i]
                # h_jointdegree_nodelist[i] = h_jointdegree_nodelist.get(i,nodecount[i][1])
                h_jointdegree_nodelist[i] = nodecount[i][1]
            elif nodecount[i][1] < nodecount[i][0]:
                # 节点数*入度>入边
                if h_jointdegree_edgecount_in[i] < i[1] * nodecount[i][0]:
                    in_unsat[i]=in_unsat.get(i,0)+i[1]*nodecount[i][0]-h_jointdegree_edgecount_in[i]
                if h_jointdegree_edgecount_out[i] < i[0] * nodecount[i][0]:
                    out_unsat[i] = out_unsat.get(i, 0) + i[0] * nodecount[i][0] - h_jointdegree_edgecount_out[i]
                # h_jointdegree_nodelist[i] = h_jointdegree_nodelist.get(i,nodecount[i][1])
                h_jointdegree_nodelist[i] = nodecount[i][1]
        elif i[0]==0:
            if h_jointdegree_edgecount_in[i]<i[1]*nodecount[i][1]:
                in_unsat[i] = in_unsat.get(i, 0) + i[1] * nodecount[i][1] - h_jointdegree_edgecount_in[i]

        elif i[1]==0:
            if h_jointdegree_edgecount_out[i] < i[0] * nodecount[i][0]:
                out_unsat[i] = out_unsat.get(i, 0) + i[0] * nodecount[i][0] - h_jointdegree_edgecount_out[i]


    # print("in_unsat",in_unsat)
    # print("out_unsat", out_unsat)
    # 修改JDM，先分别将不饱和的出度节点和入度节点相连接
    # in_sum=sum(in_unsat.values())
    # out_sum=sum(out_unsat.values())
    # print(in_sum)
    # print(out_sum)
    # while sum(in_unsat.values())>sum(out_unsat.values()):
    #     a = random.randint(1, 10)
    #     out_node = (a, 0)
    #     out_unsat[out_node]=out_unsat.get(out_node,0)+a
    #     h_jointdegree_nodelist[out_node]=h_jointdegree_nodelist.get(out_node,0)+1
    #     if sum(out_unsat.values())>sum(in_unsat.values()):
    #         out_unsat[out_node]=out_unsat.get(out_node,0)-a
    #         h_jointdegree_nodelist[out_node] = h_jointdegree_nodelist.get(out_node, 0) - 1
    #         a=sum(in_unsat.values())-sum(out_unsat.values())
    #         out_unsat[(a,0)]=out_unsat.get(a,0)+a
    #         h_jointdegree_nodelist[(a,0)] = h_jointdegree_nodelist.get((a,0), 0) + 1
    #         break
    # if in_sum>out_sum:
    while sum(out_unsat.values())>0:
        in_node=random.sample(in_unsat.keys(),1)
        # print(in_node[0])
        # print(in_unsat[in_node[0]])
        while in_unsat[in_node[0]]>0:
            # 随机在out_unsat中取一个节点
            out_node=random.sample(out_unsat.keys(),1)
            if out_unsat[out_node[0]]>0:
                JDM[(out_node[0],in_node[0])]=JDM.get((out_node[0],in_node[0]),0)+1
                out_unsat[out_node[0]]-=1
                in_unsat[in_node[0]]-=1
                if out_unsat[out_node[0]]==0:
                    del out_unsat[out_node[0]]
                    break
                if in_unsat[in_node[0]]==0:
                    del in_unsat[in_node[0]]
                    break
    # 修改JDM，然后将剩余的不饱和节点，创建新的节点与之连接 123333333333333333333333333333332222222222222
    # print(JDM)
    #
    # in_sum = sum(in_unsat.values())
    # out_sum = sum(out_unsat.values())
    # print(in_sum)
    # print(out_sum)
    # print(nodecount)



    # print("h_degree_nodelist_in", h_degree_nodelist_in)
    # print("h_degree_nodelist_out", h_degree_nodelist_out)
    # print("h_node_matchnodelist_in", h_node_matchnodelist_in)
    # print("h_node_matchnodelist_out", h_node_matchnodelist_out)
    # print("h_jointdegree_nodelist_in", h_jointdegree_nodelist_in)
    # print("h_jointdegree_nodelist_out", h_jointdegree_nodelist_out)
    # print(sum(h_jointdegree_nodelist_in.values()))
    # print(sum(h_jointdegree_nodelist_out.values()))
    # print(sum(h_jointdegree_nodelist.values()))

    # # 处理多边问题    12222222222222222222223333333333333333
    # nkk = get_nkk(JDM)
    # V={}
    # for i in V_list.keys():
    #     V[(i[0],0)]=V.get((i[0],0),0)+V_list[i]
    #     V[(i[1],1)] = V.get((i[1],1), 0) + V_list[i]
    # for k in nkk:
    #     for l in nkk[k]:
    #         edgecount = nkk[k][l]
    #         try:
    #             if V[(k, 0)] * V[(l, 1)] < edgecount:
    #                 nkk[k][l]=V[(k, 0)] * V[(l, 1)]
    #         except:
    #             print(V_out[k])
    #             print(V_in[l])

    return h_jointdegree_nodelist,JDM

def get_nkk(JDM):
    JDM_items = list(JDM.items())
    nkk={}
    for i in range(len(JDM_items)):
        # nkk.setdefault(JDM_items[i][0][0][0],{})[JDM_items[i][0][1][1]]=JDM[i]
        if JDM_items[i][1]>0:
            nkk.setdefault(JDM_items[i][0][0][0], {})
            nkk[JDM_items[i][0][0][0]].setdefault(JDM_items[i][0][1][1],0)
            nkk[JDM_items[i][0][0][0]][JDM_items[i][0][1][1]]+=JDM_items[i][1]
    return nkk

def match_as_indegree_and_outdegree(JDM,nodecount):
    JDM_items = list(JDM.items())

# p2p-Gnutella08.txt
# Wiki-Vote.txt
# example.txt
# soc-Wiki-Vote.txt
# g = nx.read_edgelist("F:\\pythonversion\\data\\p2p-Gnutella08.txt", create_using=nx.DiGraph)
# JDM=getQuintuple(g)
# print(JDM)
# in_degrees = list(dict(g.in_degree()).values())
# out_degrees = list(dict(g.out_degree()).values())
# nkk = nx.degree_mixing_dict(g)
# outdegMAX=max(out_degrees)
#
# indegMAX=max(in_degrees)
# indeg=indegree_histogram(g)
# outdeg=outdegree_histogram(g)

    # generate simple directed graph with given degree sequence and join
    # degree matrix.
