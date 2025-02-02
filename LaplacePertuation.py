import numpy as np
import networkx as nx
import D2K_simple as DK
import Graph_operation as GO
import EvaluateGraph as EG
import math
import copy
import random
#import grouping
'''
def laplace_function(x,b):
    return (1 / (2 * b)) * np.e ** (-1 * (np.abs(x) / b))


x = np.linspace(-5,5,10000)
y1 = [laplace_function(x_,0.5) for x_ in x]
y2 = [laplace_function(x_,1) for x_ in x]
y3 = [laplace_function(x_,2) for x_ in x]
'''

def noisyCount(sensitivety, epsilon):
    beta = sensitivety / epsilon
    u1 = np.random.random()
    u2 = np.random.random()
    if u1 <= 0.5:
        n_value = -beta * np.log(1. - u2)
    else:
        n_value = beta * np.log(u2)
    print(n_value)
    return n_value


def laplace_mech(data, sensitivety, epsilon):
    for i in range(len(data)):
        data[i] += noisyCount(sensitivety, epsilon)
    return data


# def getdegnode(degtype,degree,g):
#     """返回扰动后的JDAM.
#
#             Parameters
#             ----------
#             degtype:
#                 The degree type(0:indegree,1:outdegree).
#
#             degree:
#                 the specified indegree or outdegree.
#
#             g:
#                The original graph.
#
#             Returns
#             -------
#             degnode_sample:
#                 nodes list with the specified degree.
#     """
#     degnode_sample={}
#     if degtype==0:
#         d_in=g.in_degree
#         for i in g.nodes():
#             if d_in[i]==degree:
#                 degnode_sample.append(i)
#     if degtype == 1:
#         d_out = g.out_degree
#         for i in g.nodes():
#             if d_out[i] == degree:
#                 degnode_sample.append(i)
#     return degnode_sample


# def JDAMPer(outdegMAX,indegMAX,JDAM, epsilon ,sensitivety):
#     """返回扰动后的noisy_value[i][j]
#
#         Parameters
#         ----------
#         JDAM :
#             dictionary of dictionary of integers
#             directed joint degree dictionary. for nodes of out degree k (first
#             level of dict) and nodes of in degree l (seconnd level of dict)
#             describes the number of edges.
#
#         epsilon:
#            The degree type for source node (directed graphs only).
#
#         sensitivety: string ('in','out')
#            The degree type for target node (directed graphs only).
#
#
#         Returns
#         -------
#         pnkk: dictionary
#             perturbed directed joint degree dictionary
#     """
#     noisy_value = {}
#     PJDAM = copy.deepcopy(JDAM)
#     for i in JDAM:
#         for j in JDAM[i]:
#             noisy_value.setdefault(i,{})[j]=np.round(np.random.laplace(loc=0 , scale=(sensitivety/epsilon))).astype(int)
#             PJDAM[i][j] += noisy_value[i][j]
#             if PJDAM[i][j] < 0:
#                 PJDAM[i][j]=0
#                 noisy_value[i][j]=PJDAM[i][j]-JDAM[i][j]
#
#     # print("the noisy:",noisy_value)
#     # print("the perturbed jdam:",PJDAM)
#
#
#     return noisy_value



# def JDAMPer(outdegMAX,indegMAX,JDAM, epsilon ,sensitivety):
#     """返回扰动后的noisy_value[i][j]
#
#         Parameters
#         ----------
#         JDAM :
#             dictionary of dictionary of integers
#             directed joint degree dictionary. for nodes of out degree k (first
#             level of dict) and nodes of in degree l (seconnd level of dict)
#             describes the number of edges.
#
#         epsilon:
#            The degree type for source node (directed graphs only).
#
#         sensitivety: string ('in','out')
#            The degree type for target node (directed graphs only).
#
#
#         Returns
#         -------
#         pnkk: dictionary
#             perturbed directed joint degree dictionary
#     """
#     noisy_value = {}
#     PJDAM = np.zeros([outdegMAX,indegMAX])
#     for i in JDAM:
#         for j in JDAM[i]:
#             PJDAM[i][j] += JDAM[i][j]
#     for i in range(1,outdegMAX):
#         for j in range(1,indegMAX):
#             noisy_value.setdefault(i, {})[j] = np.round(np.random.laplace(loc=0, scale=(sensitivety / epsilon))).astype(int)
#             PJDAM[i][j]+=noisy_value[i][j]
#             if PJDAM[i][j] < 0:
#                 PJDAM[i][j]=0
#                 noisy_value[i][j]=0
#                 if i in JDAM and j in JDAM[i]:
#                    noisy_value[i][j]=PJDAM[i][j]-JDAM[i][j]
#
#     # print("the noisy:",noisy_value)
#     # print("the perturbed jdam:",PJDAM)
#     return noisy_value
#
# def convertJDAM(JDAM):
#     conJDAM={}
#     for i in JDAM:
#         for j in JDAM[i]:
#             conJDAM.setdefault(j, {})[i]=JDAM[i][j]
#     return conJDAM
#
# def getnk(JDAM):
#     nk_in={}
#     nk_out={}
#     tempsum=0
#     for i in JDAM:
#         for j in JDAM[i]:
#             tempsum+=JDAM[i][j]
#         nk_in[i]=np.round(tempsum/i).astype(int)
#         tempsum=0
#     conJDAM=convertJDAM(JDAM)
#     for i in conJDAM:
#         for j in conJDAM[i]:
#             tempsum+=conJDAM[i][j]
#         nk_out[i]=np.round(tempsum/i).astype(int)
#         tempsum=0
#     print("nk_out:",nk_out)
#     print("nk_in:",nk_in)
#     outdegSUM =0
#     outdegNUM=0
#     indegSUM=0
#     indegNUM=0
#     for i in nk_out:
#         outdegSUM +=i*nk_out[i]
#         outdegNUM +=nk_out[i]
#     for i in nk_in:
#         indegSUM += i * nk_in[i]
#         indegNUM += nk_in[i]
#     print("the sum of the out-degrees:",outdegSUM )
#     print("the number of the out-degrees nodes:", outdegNUM)
#     print("the sum of the in-degrees:", indegSUM)
#     print("the number of the in-degrees nodes:", indegNUM)
#     return True
#
# def getdegnode(degree,degreetype,in_degrees,out_degrees):
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

# def adjustPer(JDAM,g,noisy_value):
#     """根据扰动后的JDAM做出调整
#
#             Parameters
#             ----------
#             JDAM :
#                 dictionary of dictionary of integers
#                 directed joint degree dictionary. for nodes of out degree k (first
#                 level of dict) and nodes of in degree l (seconnd level of dict)
#                 describes the number of edges.
#
#             g:
#                the original graph
#
#             Returns
#             -------
#             in_degrees: dictionary
#                 perturbed directed joint degree dictionary
#
#         """
#
#     in_degrees = list(dict(g.in_degree()).values())
#     out_degrees = list(dict(g.out_degree()).values())
#     edgesdegree_list=list(node_degree_joint(g))
#     edgeslist = list(g.edges())
#
#     deletedges=set()
#     addedges=set()
#
#     for i in noisy_value:
#         # i为出度结点的度
#         for j in noisy_value[i]:
#             #  j为入度结点的度
#             while noisy_value[i][j]>0:
#                 #设为结点入度为j到结点出度为i之间需要新增noisy_value[i][j]条边
#                 indeg_sample = getdegnode(j, 0, in_degrees, out_degrees)
#                 outdeg_sample = getdegnode(i, 1, in_degrees, out_degrees)
#                 # if len(indeg_sample) and len(outdeg_sample):
#                     # 在节点集中随机选取节点
#                 v = random.choice(indeg_sample)
#                 w = random.choice(outdeg_sample)
#                 JDAM.setdefault(i+1,{})
#                 JDAM[i+1].setdefault(j+1,0)
#                 JDAM[i + 1][j + 1] += 1
#                 JDAM[i][j]-=1
#                 # if JDAM.get(i+1,False) and JDAM.get(i+1,{}).get(j+1,False):
#                 #     JDAM[i + 1][j + 1] +=1
#                 # else:
#                 #     JDAM.setdefault(i + 1, {})[j + 1] = 1
#                 # else:
#                 #     JDAM.setdefault(i+1, {})[j+1] = 1
#                 #     S[(k, 1)] = S.get((k, 1), 0) + val
#
#                 for m in g.predecessors(str(v)):
#                     # 对于所有以节点v为尾结点的邻居节点，JDAM[out_degrees(int(m))][j]--，JDAM[out_degrees(int(m))][j+1]++
#                     if m!= str(w):
#                         JDAM[g.out_degree(m)][j] -= 1
#                         JDAM[g.out_degree(m)].setdefault(j + 1, 0)
#                         JDAM[g.out_degree(m)][j + 1] += 1
#
#                     # if JDAM.get(g.out_degree(m),{}).get(j+1,False):
#                     #     JDAM[g.out_degree(m)][j + 1] += 1
#                     # else:
#                     #     JDAM.setdefault(g.out_degree(m), {})[j + 1] = 1
#
#                 for n in g.successors(str(w)):
#                     # 对于所有以节点w为头结点的邻居节点，JDAM[i][g.in_degree(n)]--，JDAM[i+1][g.in_degree(n)]++
#                     if n != str(v):
#                         JDAM[i][g.in_degree(n)] -= 1
#                         JDAM.setdefault(i + 1, {})
#                         JDAM[i + 1].setdefault(g.in_degree(n), 0)
#                         JDAM[i + 1][g.in_degree(n)] += 1
#
#                     # if JDAM.get(i+1, False) and JDAM.get(i+1,{}).get(g.in_degree(n),False):
#                     #     JDAM[i + 1][g.in_degree(n)] += 1
#                     # else:
#                     #     JDAM.setdefault(i+1, {})[g.in_degree(n)] = 1
#                 noisy_value[i][j] -= 1
#                 addedges.add((w,v))
#                 # else:
#                 #     nonadjust.append((i, j))
#                 #     noisy_value[i][j] -= 1
#
#
#             # 设为结点入度为j到结点出度为i之间需要减去noisy_value[i][j]条边
#             while noisy_value[i][j] < 0:
#                 edgeindex=edgesdegree_list.index((i,j))
#                 v = int(edgeslist[edgeindex][1])
#                 w = int(edgeslist[edgeindex][0])
#
#                 # outdeg_sample = getdegnode(i, 1, in_degrees, out_degrees)
#                 # indeg_sample = getdegnode(j, 0, in_degrees, out_degrees)
#                 # if len(indeg_sample) and len(outdeg_sample):
#                 # v = random.choice(indeg_sample)
#                 # w = random.choice(outdeg_sample)
#
#                 JDAM[i][j]-=1
#                 for m in g.predecessors(str(v)):
#                     # 对于所有以节点v为尾结点的邻居节点，JDAM[g.out_degree(m)][j]--,JDAM[g.out_degree(m)][j-1]++
#                     if m != str(w):
#                         JDAM[g.out_degree(m)][j] -= 1
#                         JDAM[g.out_degree(m)].setdefault(j - 1, 0)
#                         JDAM[g.out_degree(m)][j - 1] += 1
#
#                     # if JDAM.get(g.out_degree(m),{}).get(j-1,False):
#                     #     JDAM[g.out_degree(m)][j-1] += 1
#                     # else:
#                     #     JDAM.setdefault(g.out_degree(m), {})[j-1] = 1
#
#
#                 for n in g.successors(str(w)) :
#                     # 对于所有以节点w为头结点的邻居节点，JDAM[i][g.in_degree(n)]--,JDAM[i-1][g.in_degree(n)]--
#                     if n != str(v):
#                         JDAM[i][g.in_degree(n)] -= 1
#                         JDAM.setdefault(i - 1, {})
#                         JDAM[i - 1].setdefault(g.in_degree(n), 0)
#                         JDAM[i - 1][g.in_degree(n)] += 1
#                         # if JDAM.get(i-1, False) and JDAM.get(i-1,{}).get(g.in_degree(n),False):
#                         #     JDAM[i - 1][g.in_degree(n)] += 1
#                         # else:
#                         #     JDAM.setdefault(i-1, {})[g.in_degree(n)] = 1
#
#                 noisy_value[i][j] += 1
#                 deletedges.add((w, v))
#                 # else:
#                 #     nonadjust.append((i, j))
#                 #     noisy_value[i][j] += 1
#
#     return JDAM

# def adjustPer(g,noisy_value):
#     """根据扰动后的JDAM做出调整
#
#             Parameters
#             ----------
#             JDAM :
#                 dictionary of dictionary of integers
#                 directed joint degree dictionary. for nodes of out degree k (first
#                 level of dict) and nodes of in degree l (seconnd level of dict)
#                 describes the number of edges.
#
#             g:
#                the original graph
#
#             Returns
#             -------
#             in_degrees: dictionary
#                 perturbed directed joint degree dictionary
#
#         """
#
#     in_degrees = list(dict(g.in_degree()).values())
#     out_degrees = list(dict(g.out_degree()).values())
#     edgesdegree_list=list(node_degree_joint(g))
#     edgeslist = list(g.edges())
#
#     deletedges=set()
#     addedges=set()
#
#     for i in noisy_value:
#         # i为出度结点的度
#         for j in noisy_value[i]:
#             #  j为入度结点的度
#             while noisy_value[i][j]>0:
#                 #设为结点入度为j到结点出度为i之间需要新增noisy_value[i][j]条边
#                 indeg_sample = getdegnode(j, 0, in_degrees, out_degrees)
#                 outdeg_sample = getdegnode(i, 1, in_degrees, out_degrees)
#                 # if len(indeg_sample) and len(outdeg_sample):
#                     # 在节点集中随机选取节点
#                 v = random.choice(indeg_sample)
#                 w = random.choice(outdeg_sample)
#                 noisy_value[i][j] -= 1
#                 addedges.add((w,v))
#
#
#
#             # 设为结点入度为j到结点出度为i之间需要减去noisy_value[i][j]条边
#             while noisy_value[i][j] < 0:
#                 edgeindex=edgesdegree_list.index((i,j))
#                 v = int(edgeslist[edgeindex][1])
#                 w = int(edgeslist[edgeindex][0])
#                 noisy_value[i][j] += 1
#                 deletedges.add((w, v))
#
#     intersection=deletedges & addedges
#     deletedges=deletedges-intersection
#     addedges = addedges - intersection
#     g.add_edges_from(list(addedges))
#     g.remove_edges_from(list(deletedges))
#
#     return g



# def test_Pertuate(g,epsilon):
#
#
#     in_degrees = list(dict(g.in_degree()).values())
#     out_degrees = list(dict(g.out_degree()).values())
#     outdegMAX = max(out_degrees)
#     indegMAX = max(in_degrees)
#
#     JDAM = nx.degree_mixing_dict(g)
#     # print("the original JDAM", JDAM)
#     # print("未添加噪声的生成图统计特性如下：")
#     # PG=DK.directed_joint_degree_model(in_degrees, out_degrees, JDAM)
#     # DK.EvaluateGraph.evaluate(PG)
#     sensitivety = outdegMAX + indegMAX + 1
#     epsilon = epsilon
#
#     noisy_value=JDAMPer(outdegMAX, indegMAX, JDAM, epsilon, sensitivety)
#     G = adjustPer(g, noisy_value)
#     adJDAM = nx.degree_mixing_dict(G)
#     # print("the adjusted PerJDAM:", adJDAM)
#     new_in_degrees = list(dict(G.in_degree()).values())
#     new_out_degrees = list(dict(G.out_degree()).values())
#
#     GEN_G = DK.directed_joint_degree_model(new_in_degrees, new_out_degrees, adJDAM)
#     print("添加噪声后的生成图统计特性如下(此时的epsilon为：",epsilon,"):")
#     DK.EvaluateGraph.evaluate(GEN_G)

# in_degrees = list(dict(g.in_degree()).values())
# out_degrees = list(dict(g.out_degree()).values())
# print("原图的统计特性如下：")
# DK.EvaluateGraph.evaluate(g)
# JDAM = nx.degree_mixing_dict(g)
#
# print("未添加噪声的生成图统计特性如下：")
# PG=DK.directed_joint_degree_model(in_degrees, out_degrees, JDAM)
# DK.EvaluateGraph.evaluate(PG)

# test_Pertuate(g,30)
# test_Pertuate(g,15)
# test_Pertuate(g,10)
# test_Pertuate(g,5)
# test_Pertuate(g,50)


# def per_JDM(JDM,epsilon ,sensitivety):
#     noisy_value = []
#     JDM_values=list(JDM.values())
#     NewJDM=[]
#     for i in range(len(JDM_values)):
#         noisy_value.append(np.round(np.random.laplace(loc=0, scale=(sensitivety / epsilon))).astype(int))
#         NewJDM.append(noisy_value[i]+JDM_values[i])
#         if NewJDM[i]<0:
#             NewJDM[i]=0
#             noisy_value[i]=NewJDM[i]-JDM_values[i]
#
#     return NewJDM
def per_JDM(JDM,epsilon):
    maxIndegree=0
    maxOutdegree=0
    DpJDM_value=[]
    numOfZero = 0
    valsLessThanZero = 0
    JDM_items = list(JDM.items())
    for i in range(len(JDM_items)):
        if(JDM_items [i][0][0][0]>maxOutdegree):
            maxOutdegree=JDM_items [i][0][0][0]
        if(JDM_items [i][0][1][0]>maxOutdegree):
            maxOutdegree = JDM_items [i][0][1][0]
        if (JDM_items [i][0][0][1] > maxIndegree):
            maxIndegree = JDM_items [i][0][0][1]
        if (JDM_items [i][0][1][1] > maxIndegree):
            maxIndegree = JDM_items [i][0][1][1]
    sensitivity=maxOutdegree+maxIndegree
    absPositiveSum = 0
    absNegativeSum = 0
    while (absNegativeSum >= absPositiveSum):
        JDM_values = list(JDM.values())
        Laplacenoisy = np.random.laplace(loc=0, scale=(1 / epsilon), size=len(JDM_values))
        print(Laplacenoisy)
        DpJDM_value=JDM_values+Laplacenoisy
        print(DpJDM_value)
        for i in range(len(DpJDM_value)):
            newVal=DpJDM_value[i]
            if  newVal<0:
                absNegativeSum+=abs(newVal)
                valsLessThanZero+=-1 * newVal
                numOfZero+=1
                DpJDM_value[i]=0
                newVal=0
            else:
                DpJDM_value[i]=np.round(newVal)
                absPositiveSum+=newVal
            JDM[JDM_items [i][0]]=np.round(newVal)
    iterSum = 0
    cutValue = 0
    iterCount = 0
    for i in range(len(DpJDM_value)):
        iterSum+=((DpJDM_value[i]-cutValue)*(len(DpJDM_value)-iterCount))
        if(iterSum>absNegativeSum):
            break
        cutValue=DpJDM_value[i]
        iterCount+=1
    if iterCount!=len(DpJDM_value):
        sub=(absNegativeSum - iterSum) / (len(DpJDM_value) - iterCount)
        for i in JDM.keys():
            if JDM[i]<=cutValue:
                JDM[i]=JDM[i]-cutValue-sub
    return JDM







def reconstruct_nodejoin(JDM,NewJDM):

    node_list = {}

    # JDM_items = list(JDM.items())
    # # print("the JDM_items:")
    # # print(JDM_items[0])
    # for i in range(len(JDM_items)):
    #     node_list[JDM_items[i][0][0]]= node_list.get((JDM_items[i][0][0]), 0)+np.round(math.ceil(NewJDM[i]/JDM_items[i][0][0][0])).astype(int)
    #     node_list[JDM_items[i][0][1]] = node_list.get((JDM_items[i][0][1]), 0) + np.round(math.ceil(NewJDM[i]/JDM_items[i][0][1][1])).astype(int)
    # return node_list

def getIndegreeSeq(node_list):
    IndegreeSeq=[]
    nodelist=list(node_list.items())
    for i in range(len(nodelist)):
        if nodelist[i]==0:
            break
        for j in range(0,nodelist[i][1]):
            IndegreeSeq.append(nodelist[i][0][1])
    return IndegreeSeq

def getOutdegreeSeq(node_list):
    OutdegreeSeq=[]
    nodelist = list(node_list.items())
    for i in range(len(nodelist)):
        if nodelist[i]==0:
            break
        for j in range(0,nodelist[i][1]):
            OutdegreeSeq.append(nodelist[i][0][0])
    return OutdegreeSeq

# def get_nkk(JDM,NewJDM):
#     JDM_items = list(JDM.items())
#     nkk={}
#     for i in range(len(JDM_items)):
#         # nkk.setdefault(JDM_items[i][0][0][0],{})[JDM_items[i][0][1][1]]=NewJDM[i]
#         nkk.setdefault(JDM_items[i][0][0][0], {})
#
#         nkk[JDM_items[i][0][0][0]][JDM_items[i][0][1][1]] =nkk[JDM_items[i][0][0][0]].get(JDM_items[i][0][1][1],0)+NewJDM[i]
#     return nkk

def remove_zero(JDM,NewJDM):
    JDM_keys = list(JDM.keys())
    JDM_values=list(JDM.values())
    tempNewJDM=[]
    tempJDM={}
    for i in range(len(NewJDM)):
        if NewJDM[i] != 0:
            tempNewJDM.append(NewJDM[i])
            tempJDM.setdefault(JDM_keys[i],JDM_values[i])
    JDM.clear()
    JDM=tempJDM
    NewJDM.clear()
    NewJDM=tempNewJDM
    return 0

def getNew_JDM(JDM,NewJDM):
    JDM_keys=list(JDM.keys())
    for i in range(len(NewJDM)):
        JDM[JDM_keys[i]]=NewJDM[i]
    return JDM

# print(g)
# EG.evaluate(g)
# JDM=GO.getQuintuple(g)
# # JDM1、2、3分别存储度之和deg<=40、40<deg<108、deg>=108的边
# JDM_items = list(JDM.items())
# JDM_keys=list(JDM.keys())
# JDM_values=list(JDM.values())
# JDM1={}
# JDM2={}
# JDM3={}
# for i in range(len(JDM_items)):
#     degsum=JDM_items[i][0][0][0]+JDM_items[i][0][0][1]+JDM_items[i][0][1][0]+JDM_items[i][0][1][1]
#     if degsum<=40:
#         JDM1.setdefault(JDM_keys[i], JDM_values[i])
#     if degsum>40 and degsum<108:
#         JDM2.setdefault(JDM_keys[i], JDM_values[i])
#     if degsum>=108:
#         JDM3.setdefault(JDM_keys[i], JDM_values[i])
# # print(JDM1)
# # print(JDM2)
# # print(JDM3)
#
# NewJDM1=per_JDM(JDM1,5,40)
# NewJDM2=per_JDM(JDM2,5,108)
# NewJDM3=per_JDM(JDM3,5,200)
#
#
# # remove_zero(JDM1,NewJDM1)
# tempNewJDM=[]
# tempJDM={}
# JDM_values=list(JDM1.values())
# JDM_keys=list(JDM1.keys())
# for i in range(len(NewJDM1)):
#     if NewJDM1[i] != 0:
#         tempNewJDM.append(NewJDM1[i])
#         tempJDM.setdefault(JDM_keys[i],JDM_values[i])
# JDM1.clear()
# JDM1=tempJDM
# NewJDM1.clear()
# NewJDM1=tempNewJDM
# tempNewJDM=[]
# tempJDM={}
# # remove_zero(JDM2,NewJDM2)
# JDM_values=list(JDM2.values())
# JDM_keys=list(JDM2.keys())
# for i in range(len(NewJDM2)):
#     if NewJDM2[i] != 0:
#         tempNewJDM.append(NewJDM2[i])
#         tempJDM.setdefault(JDM_keys[i],JDM_values[i])
# JDM2.clear()
# JDM2=tempJDM
# NewJDM2.clear()
# NewJDM2=tempNewJDM
# tempNewJDM=[]
# tempJDM={}
# # remove_zero(JDM3,NewJDM3)
# JDM_values=list(JDM3.values())
# JDM_keys=list(JDM3.keys())
# for i in range(len(NewJDM3)):
#     if NewJDM3[i] != 0:
#         tempNewJDM.append(NewJDM3[i])
#         tempJDM.setdefault(JDM_keys[i],JDM_values[i])
# JDM3.clear()
# JDM3=tempJDM
# NewJDM3.clear()
# NewJDM3=tempNewJDM
# # print(JDM1)
# # print(JDM2)
# # print(JDM3)
#
# AllJDM={**JDM1,**JDM2,**JDM3}
# AllNewJDM=NewJDM1+NewJDM2+NewJDM3
# node_list=reconstruct_nodejoin(AllJDM,AllNewJDM)
# newIndegSeq=getIndegreeSeq(node_list)
# newOutdegSeq=getOutdegreeSeq(node_list)
# newNkk=get_nkk(AllJDM,AllNewJDM)
# print("the new Indegree Sequence:")
# print(newIndegSeq)
# print("the new Outdegree Sequence:")
# print(newOutdegSeq)
# print("the new nkk:")
# print(newNkk)
# print(DK.is_valid_directed_joint_degree(newIndegSeq, newOutdegSeq,newNkk))

def groupingPer(g,sortlist,epsilon):
    JDM = GO.getQuintuple(g)
    print(JDM)
    JDM_items = list(JDM.items())
    JDM_keys = list(JDM.keys())
    JDM_values = list(JDM.values())
    JDM_group = {}
    n=len(sortlist)

    # 进行分组
    for i in range(0,n):
        JDM_group[i]={}
    for i in range(len(JDM_items)):
        degsum = JDM_items[i][0][0][0] + JDM_items[i][0][0][1] + JDM_items[i][0][1][0] + JDM_items[i][0][1][1]
        tempn=0
        while(tempn<n):
            if degsum<sortlist[tempn]:
                JDM_group[tempn].setdefault(JDM_keys[i], JDM_values[i])
                tempn=n+1
            else:
                tempn+=1
    # 分组进行扰动
    NewJDMgroup=[[]for i in range(0,n)]
    for i in range(0, n):
        NewJDMgroup[i]=per_JDM(JDM_group[i],epsilon,sortlist[i])

    # 扰动后，将扰动为0的项去掉
    for i in range(0, n):
        tempNewJDM = []
        tempJDM = {}
        jdm_values = list(JDM_group[i].values())
        jdm_keys = list(JDM_group[i].keys())
        for j in range(len(NewJDMgroup[i])):
            if NewJDMgroup[i][j] != 0:
                tempNewJDM.append(NewJDMgroup[i][j])
                tempJDM.setdefault(jdm_keys[j], jdm_values[j])
        JDM_group[i].clear()
        NewJDMgroup[i].clear()
        JDM_group[i] = copy.deepcopy(tempJDM)
        NewJDMgroup[i] = copy.deepcopy(tempNewJDM)



    AllJDM={}
    AllNewJDM=[]
    m=0

    for i in range(0, n):
        # for j in range(len(JDM_group[i])):
        #     AllJDM[m]=JDM_group[i][j]

        AllJDM.update(JDM_group[i])
        AllNewJDM+=NewJDMgroup[i]


    New_JDM = getNew_JDM(AllJDM, AllNewJDM)

    node_list,New_JDM = Adapt(New_JDM)
    newIndegSeq = getIndegreeSeq(node_list)
    newOutdegSeq = getOutdegreeSeq(node_list)
    newNkk = GO.get_nkk(New_JDM)
    G = DK.directed_joint_degree_model(newIndegSeq, newOutdegSeq, newNkk)
    # print(G)
    EG.evaluate(G)
    return G

def regularper(JDM,epsilon):
    # JDM = GO.getQuintuple(g)
    # 第一步，对JDM进行扰动
    NewJDM=per_JDM(JDM,epsilon)
    JDM_keys = list(JDM.keys())
    JDM_values = list(JDM.values())
    AllJDM = {}
    AllNewJDM = []
    # 去除扰动后的JDM中，为0的数据项
    for i in range(len(NewJDM)):
        if NewJDM[i] != 0:
            AllNewJDM.append(NewJDM[i])
            AllJDM.setdefault(JDM_keys[i], JDM_values[i])
    New_JDM=getNew_JDM(AllJDM, AllNewJDM)
    # print(New_JDM)
    # 第三步，根据JDM调整，生成匹配生成图算法的输出
    node_list,New_JDM = adapt(New_JDM)
    # print(node_list)
    newIndegSeq = getIndegreeSeq(node_list)
    newOutdegSeq = getOutdegreeSeq(node_list)
    newNkk = GO.get_nkk(New_JDM)

    DK.is_valid_directed_joint_degree(newIndegSeq, newOutdegSeq, newNkk)
    G = DK.directed_joint_degree_model(newIndegSeq, newOutdegSeq, newNkk)
    # print(G)
    EG.evaluate(G)
    return G

def Adapt(JDM):
    """对JDM加噪声之后，适配算法要解决的几个问题：
    case1：随机加噪后的JDM，从中提取出来的出度序列之和与入度序列之和不一致，导致同一个节点从出度序列提取到的数量与从入度序列提取到的不一致（不符合有向图的性质）

    case2：随机加噪后的JDM,提取到的节点数量后仍存在不饱和边（不符合有向图的性质）

    case3：随机加噪后，出现多边（D2K-Simple生成图算法不支持该情况）的问题

    :param JDM:
    :return: V_list,JDM
    """
    # 存储每种度所占的边数
    S_in = {}
    S_out = {}
    # 存储每种度应该要有的节点数
    V_in = {}
    V_out = {}
    # 存储每种联合节点应该要有的数量
    V_list = {}

    # 临时存储每种度应该要有的节点数，对应D2K-Simple中的V = {}  # number of nodes with in/out degree.
    nodecount = {}
    # 对于不饱和的节点，以度和缺边数进行记录
    out_unsat = {}
    in_unsat = {}
    JDM_items = list(JDM.items())

    # 计算每种度所占的边数
    for i in range(len(JDM_items)):
        S_in[JDM_items[i][0][1]] = S_in.get(JDM_items[i][0][1], 0) + JDM_items[i][1]
        if JDM_items[i][0][1][0]!=0:
            S_out[JDM_items[i][0][1]] = S_out.get(JDM_items[i][0][1], 0)
        S_out[JDM_items[i][0][0]] = S_out.get(JDM_items[i][0][0], 0) + JDM_items[i][1]
        if JDM_items[i][0][0][1]!=0:
            S_in[JDM_items[i][0][0]]=S_in.get(JDM_items[i][0][0], 0)

    # 计算每种入度的节点个数，V_in[k]/k,同时在nodecount = {}中记录入度为k的节点个数，记录形式为nodecount[(k,1)]=number
    for i in S_in.keys():
        V_in.setdefault(i[1],{})
        V_in[i[1]][i] = V_in[i[1]].get(i, 0) + math.ceil(
            S_in[i] / i[1])
        nodecount.setdefault(i, {})
        nodecount[i][1] = V_in[i[1]][i]
        nodecount[i][0]=nodecount[i].get(0,0)

    # 计算每种出度的节点个数，V_out[k]/k，同时在nodecount = {}中记录入度为k的节点个数，记录形式为nodecount[(k,0)]=number
    for i in S_out.keys():
        V_out.setdefault(i[0],{})
        V_out[i[0]][i] = V_out[i[0]].get(i, 0) + math.ceil(
            S_out[i] / i[0])
        nodecount.setdefault(i, {})
        nodecount[i][0] = V_out[i[0]][i]
        nodecount[i][1] = nodecount[i].get(1,0)

    # 检测多边问题
    for i in range(len(JDM_items)):
        in_node=JDM_items[i][0][1]
        out_node=JDM_items[i][0][0]
        if JDM_items[i][1]>nodecount[in_node][1]*nodecount[out_node][0]:
            JDM[(out_node, in_node)]=nodecount[in_node][1]*nodecount[out_node][0]
            out_unsat[out_node] = out_unsat.get(out_node, 0) + JDM_items[i][1]-nodecount[in_node][1]*nodecount[out_node][0]
            in_unsat[in_node]=in_unsat.get(in_node,0)+JDM_items[i][1]-nodecount[in_node][1]*nodecount[out_node][0]
    # print(sum(out_unsat.values()))
    # print(sum(in_unsat.values()))
    # print("JDM1", JDM)
    # print("nodecount1:", nodecount)
    # print("out_unsat1:", out_unsat)
    # print("in_unsat1:", in_unsat)

    # 考虑以下情况：某节点v(outdeg,indeg),由outdeg计算节点个数为x个，由indeg计算节点个数为y个，需要统一为大的那个 并且进一步处理
    for i in nodecount.keys():
        # 如果节点仅为出度节点
        if i[0] == 0:
            if S_in[i] < i[1] * nodecount[i][1]:
                in_unsat[i] = in_unsat.get(i, 0) + i[1] * nodecount[i][1] - S_in[i]
        # 如果节点仅为入度节点
        elif i[1] == 0:
            if S_out[i] < i[0] * nodecount[i][0]:
                out_unsat[i] = out_unsat.get(i, 0) + i[0] * nodecount[i][0] - S_out[i]
        elif i[0] != 0 and i[1] != 0:
            # 处理case1：如果入度计算的节点个数大于出度计算的节点个数，统一成入度节点个数，并添加出度
            if nodecount[i][1] > nodecount[i][0]:
                # 处理case2，如果节点数*入度>入边，意味着入度为i的节点还未饱和，记录下来
                nodecount[i][0] = nodecount[i][1]
                if S_in[i] < i[1] * nodecount[i][1]:
                    in_unsat[i] = in_unsat.get(i, 0) + i[1] * nodecount[i][1] - S_in[i]
                if S_out[i] < i[0] * nodecount[i][1]:
                    out_unsat[i] = out_unsat.get(i, 0) + i[0] * nodecount[i][1] - S_out[i]

            # 处理case1：如果入度计算的节点个数小于出度计算的节点个数，统一成出度节点个数，并添加出度
            elif nodecount[i][1] < nodecount[i][0]:
                nodecount[i][1] = nodecount[i][0]
                # 处理case2，如果节点数*出度>出边，意味着出度为i的节点还未饱和，记录下来
                if S_in[i] < i[1] * nodecount[i][0]:
                    in_unsat[i] = in_unsat.get(i, 0) + i[1] * nodecount[i][0] - S_in[i]
                if S_out[i] < i[0] * nodecount[i][0]:
                    out_unsat[i] = out_unsat.get(i, 0) + i[0] * nodecount[i][0] - S_out[i]
            else:
                if S_in[i] < i[1] * nodecount[i][1]:
                    in_unsat[i] = in_unsat.get(i, 0) + i[1] * nodecount[i][1] - S_in[i]
                if S_out[i] < i[0] * nodecount[i][0]:
                    out_unsat[i] = out_unsat.get(i, 0) + i[0] * nodecount[i][0] - S_out[i]
    # print("nodecount2:", nodecount)
    # print("out_unsat2:", out_unsat)
    # print("in_unsat2:", in_unsat)

    """
    解决不饱和问题，随机连接不饱和点
    """
    # 添加虚拟节点，修改节点集，使出、入度不饱和的边数相等
    diff=sum(out_unsat.values())-sum(in_unsat.values())
    if diff>0:
        in_unsat[(0, 1)] = in_unsat.get((0, 1), 0) + diff
        nodecount.setdefault((0,1),{})
        nodecount[(0,1)][1]=nodecount[(0,1)].get(1,0)+diff
    elif diff<0:
        out_unsat[(1,0)] = out_unsat.get((1, 0), 0) + abs(diff)
        nodecount.setdefault((1,0), {})
        nodecount[(1,0)][0] = nodecount[(1,0)].get(0, 0) +abs(diff)

    # 随机连接不饱和节点,添加边，修改相应的JDM
    flagsum_out=sum(out_unsat.values())
    while sum(out_unsat.values())>0:
        in_node=random.sample(in_unsat.keys(),1)
        # 如果没有满足要求的节点选取
        if flagsum_out <= 0:
            break
        # print(in_node[0])
        # print(in_unsat[in_node[0]])
        while in_unsat[in_node[0]]>0:
            # 随机在out_unsat中取一个节点
            out_node=random.sample(out_unsat.keys(),1)
            # 处理out_unsat、in_unsat中取出来的节点为同一节点的情况
            if out_node[0]!=in_node[0]:
                if JDM.get((out_node[0],in_node[0]),0)>=nodecount[in_node[0]][1]*nodecount[out_node[0]][0]:
                    flagsum_out-=1
                    break
                elif out_unsat[out_node[0]] > 0 :
                    JDM[(out_node[0], in_node[0])] = JDM.get((out_node[0], in_node[0]), 0) + 1
                    out_unsat[out_node[0]] -= 1
                    in_unsat[in_node[0]] -= 1
                    if out_unsat[out_node[0]] == 0:
                        del out_unsat[out_node[0]]
                        flagsum_out -= 1
                    if in_unsat[in_node[0]] == 0:
                        del in_unsat[in_node[0]]
                        break
            elif out_node[0]==in_node[0]:
                if JDM.get((out_node[0],in_node[0]),0)>=(nodecount[in_node[0]][1]-1)*(nodecount[out_node[0]][0]-1):
                    flagsum_out -= 1
                    break
                elif out_unsat[out_node[0]] > 0 :
                    JDM[(out_node[0], in_node[0])] = JDM.get((out_node[0], in_node[0]), 0) + 1
                    out_unsat[out_node[0]] -= 1
                    in_unsat[in_node[0]] -= 1
                    if out_unsat[out_node[0]] == 0:
                        del out_unsat[out_node[0]]
                        flagsum_out -= 1
                    if in_unsat[in_node[0]] == 0:
                        del in_unsat[in_node[0]]
                        break
    # print("JDM3",JDM)
    # print("nodecount3:", nodecount)
    # print("out_unsat3:", out_unsat)
    # print("in_unsat3:", in_unsat)
    while sum(out_unsat.values()) > 0:
        out_node = random.sample(out_unsat.keys(), 1)
        JDM[(out_node[0], (0,1))]= JDM.get((out_node[0], (0,1)), 0)+1
        out_unsat[out_node[0]] -= 1
        if out_unsat[out_node[0]] == 0:
            del out_unsat[out_node[0]]
        nodecount.setdefault((0, 1), {})
        nodecount[(0, 1)][1] = nodecount[(0, 1)].get(1, 0) + 1
    while sum(in_unsat.values()) > 0:
        in_node = random.sample(in_unsat.keys(), 1)
        JDM[((1,0), in_node[0])]= JDM.get(((1,0), in_node[0]), 0)+1
        in_unsat[in_node[0]] -= 1
        if in_unsat[in_node[0]] == 0:
            del in_unsat[in_node[0]]
        nodecount.setdefault((1, 0), {})
        nodecount[(1,0)][0] = nodecount[(1,0)].get(0, 0) +1
    # print("JDM4", JDM)
    # print("nodecount4:",nodecount)
    # print("out_unsat4:", out_unsat)
    # print("in_unsat4:", in_unsat)
    print("JDM Adapt successfullly")


    for i in nodecount.keys():
        # 如果节点仅为出度节点
        if i[0]==0:
            V_list[i]=V_list.get(i,nodecount[i][1])
        # 如果节点仅为入度节点
        elif i[1]==0:
            V_list[i]=V_list.get(i,nodecount[i][0])
        else :
            V_list[i]=V_list.get(i,nodecount[i][0])
    # print(V_list)

    return V_list, JDM
def adapt(JDM):

    # 存储可能的度
    V = []
    # 存储每种度所占的边数,以及对应的联合度
    S_in = {}
    S_out = {}
    # 存储每种联合节点应该要有的数量
    V_list = {}

    # 临时存储每种度应该要有的节点数，对应D2K-Simple中的V = {}  # number of nodes with in/out degree.
    nodecount = {}
    # 对于不饱和的节点，以度和缺边数进行记录
    out_unsat = {}
    in_unsat = {}
    JDM_items = list(JDM.items())
    # 计算每种度所占的边数

    for i in range(len(JDM_items)):
        if JDM_items[i][0][1][1] != 0:
            S_in[JDM_items[i][0][1]] = S_in.get(JDM_items[i][0][1], {})
        if JDM_items[i][0][1][0] != 0:
            S_out[JDM_items[i][0][1]] = S_out.get(JDM_items[i][0][1], {})
        if JDM_items[i][0][0][1] != 0:
            S_in[JDM_items[i][0][0]] = S_in.get(JDM_items[i][0][0], {})
        if JDM_items[i][0][0][0] != 0:
            S_out[JDM_items[i][0][0]] = S_out.get(JDM_items[i][0][0], {})

        S_in[JDM_items[i][0][1]][JDM_items[i][0][0]] = S_in[JDM_items[i][0][1]].get(JDM_items[i][0][0], 0) + \
                                                       JDM_items[i][1]
        S_out[JDM_items[i][0][0]][JDM_items[i][0][1]] = S_out[JDM_items[i][0][0]].get(JDM_items[i][0][1], 0) + \
                                                        JDM_items[i][1]
        V.append(JDM_items[i][0][1])
        V.append(JDM_items[i][0][0])
    V = list(set(V))
    print(V)

    # 计算每种入度的节点个数，V_in[k]/k,同时在nodecount = {}中记录入度为k的节点个数，记录形式为nodecount[(k,1)]=number
    for i in S_in.keys():
        nodecount.setdefault(i, {})
        nodecount[i][1] = round(sum(list(S_in[i].values())) / i[1], 4)
        nodecount[i][0] = nodecount[i].get(0, 0)

    # 计算每种出度的节点个数，V_out[k]/k，同时在nodecount = {}中记录入度为k的节点个数，记录形式为nodecount[(k,0)]=number
    for i in S_out.keys():
        nodecount.setdefault(i, {})
        nodecount[i][0] = round(sum(list(S_out[i].values())) / i[0], 4)
        nodecount[i][1] = nodecount[i].get(1, 0)

    for i in range(len(V)):
        # 对单出入度节点进行处理
        if V[i][0] == 0:  # 如果是入度节点
            # 如果入度节点数为整数
            if nodecount[V[i]][1] % 1 == 0:
                nodecount[V[i]][1] = int(nodecount[V[i]][1])
                V_list[V[i]] = V_list.get(V[i], 0) + int(nodecount[V[i]][1])
                newindeg = 0
            # 如果入度节点数为小于1.5
            elif nodecount[V[i]][1] < 1.5:
                # 获取新的入度替代
                newindeg = sum(list(S_in[V[i]].values())) - V_list.get(V[i], 0) * V[i][1]
                V_list[(0, newindeg)] = V_list.get((0, newindeg), 0) + 1
            # 如果入度节点数为大于1.5
            else:
                # 如果入度节点数的小数部分全都小于0.5
                if math.modf(nodecount[V[i]][1])[0] < 0.5:
                    # 更新nodecount
                    V_list[V[i]] = V_list.get(V[i], 0) + int(nodecount[V[i]][1]) - 1
                    nodecount[V[i]][1] = int(nodecount[V[i]][1]) - 1
                # 如果入度节点数的小数部分全都大于0.5
                if math.modf(nodecount[V[i]][1])[0] >= 0.5:
                    # 更新nodecount
                    V_list[V[i]] = V_list.get(V[i], 0) + int(nodecount[V[i]][1])
                    nodecount[V[i]][1] = int(nodecount[V[i]][1])

                newindeg = sum(list(S_in[V[i]].values())) - V_list.get(V[i], 0) * V[i][1]
                V_list[(0, newindeg)] = V_list.get((0, newindeg), 0) + 1
            # 更新JDM
            j = 0
            while j < newindeg:
                outnode_choice = random.sample(S_in[V[i]].keys(), 1)
                outnode = outnode_choice[0]
                if S_out[outnode][V[i]] > 0:
                    # 更新JDM
                    JDM[(outnode, (0, newindeg))] = JDM.get((outnode, (0, newindeg)), 0) + 1
                    JDM[(outnode, V[i])] = JDM[(outnode, V[i])] - 1
                    # 更新S_in、S_out

                    S_out[outnode][(0, newindeg)] = S_out[outnode].get((0, newindeg), 0) + 1
                    S_out[outnode][V[i]] -= 1

                    S_in[(0, newindeg)] = S_in.get((0, newindeg), {})
                    S_in[(0, newindeg)][outnode] = S_in[(0, newindeg)].get(outnode, 0) + 1
                    S_in[V[i]][outnode] -= 1

                    j += 1
                else:
                    continue

        elif V[i][1] == 0:  # 如果是出度节点
            # 如果出度节点数为整数
            if nodecount[V[i]][0] % 1 == 0:
                nodecount[V[i]][0] = int(nodecount[V[i]][0])
                V_list[V[i]] = V_list.get(V[i], 0) + int(nodecount[V[i]][0])
                newoutdeg = 0
            # 如果出度节点数小于1.5
            elif nodecount[V[i]][0] < 1.5:
                newoutdeg = sum(list(S_out[V[i]].values())) - V_list.get(V[i], 0) * V[i][0]
                # 更新V_list
                V_list[(newoutdeg, 0)] = V_list.get((newoutdeg, 0), 0) + 1
            # 如果出度节点数大于1.5
            else:
                # 如果出度节点数的小数部分全都小于0.5
                if math.modf(nodecount[V[i]][0])[0] < 0.5:
                    # 更新nodecount
                    V_list[V[i]] = V_list.get(V[i], 0) + int(nodecount[V[i]][0]) - 1
                    nodecount[V[i]][0] = int(nodecount[V[i]][0]) - 1
                # 如果出度节点数的小数部分全都大于0.5
                if math.modf(nodecount[V[i]][0])[0] >= 0.5:
                    # 更新nodecount
                    V_list[V[i]] = V_list.get(V[i], 0) + int(nodecount[V[i]][0])
                    nodecount[V[i]][0] = int(nodecount[V[i]][0])

                newoutdeg = sum(list(S_out[V[i]].values())) - V_list.get(V[i], 0) * V[i][0]
                V_list[(newoutdeg, 0)] = V_list.get((newoutdeg, 0), 0) + 1
            # 更新JDM
            j = 0
            while j < newoutdeg:
                innode_choice = random.sample(S_out[V[i]].keys(), 1)
                innode = innode_choice[0]
                if S_out[V[i]][innode] > 0:
                    # 更新JDM
                    JDM[((newoutdeg, 0), innode)] = JDM.get(((newoutdeg, 0), innode), 0) + 1
                    JDM[(V[i], innode)] = JDM[(V[i], innode)] - 1
                    # 更新S_in、S_out

                    S_out[(newoutdeg, 0)] = S_out.get((newoutdeg, 0), {})
                    S_out[(newoutdeg, 0)][innode] = S_out[(newoutdeg, 0)].get(innode, 0) + 1
                    S_out[V[i]][innode] -= 1

                    S_in[innode][(newoutdeg, 0)] = S_in[innode].get((newoutdeg, 0), 0) + 1
                    S_in[innode][V[i]] = S_in[innode][V[i]] - 1
                    j += 1
                else:
                    continue
        else:  # 如果为联合度节点
            # 如果联合度节点的出度节点个数和入度节点个数相等且为整数
            if nodecount[V[i]][0] % 1 == 0 and nodecount[V[i]][1] % 1 == 0 and nodecount[V[i]][0] == nodecount[V[i]][1]:
                nodecount[V[i]][0] = int(nodecount[V[i]][0])
                V_list[V[i]] = V_list.get(V[i], 0) + int(nodecount[V[i]][1])
                newoutdeg = 0
                newindeg = 0
            # 如果联合度节点的出度节点个数和入度节点个数都小于1.5
            elif nodecount[V[i]][0] < 1.5 or nodecount[V[i]][1] < 1.5:
                newoutdeg = sum(list(S_out[V[i]].values())) - V_list.get(V[i], 0) * V[i][0]
                newindeg = sum(list(S_in[V[i]].values())) - V_list.get(V[i], 0) * V[i][1]
                V_list[(newoutdeg, newindeg)] = V_list.get((newoutdeg, newindeg), 0) + 1
            # 如果联合度节点的出度节点个数和入度节点个数有一个不小于1.5
            else:
                integer_out = int(nodecount[V[i]][0])
                integer_in = int(nodecount[V[i]][1])
                decimal_out = nodecount[V[i]][0]
                decimal_in = nodecount[V[i]][1]
                V_list[V[i]] = V_list.get(V[i], 0) + min(integer_in, integer_out) - 1
                nodecount[V[i]][0] = min(integer_in, integer_out) - 1
                nodecount[V[i]][1] = min(integer_in, integer_out) - 1
                decimal_out = decimal_out - nodecount[V[i]][0]
                decimal_in = decimal_in - nodecount[V[i]][1]
                if decimal_out < 1.5 or decimal_in < 1.5:
                    newindeg = sum(list(S_in[V[i]].values())) - V_list.get(V[i], 0) * V[i][1]
                    newoutdeg = sum(list(S_out[V[i]].values())) - V_list.get(V[i], 0) * V[i][0]
                else:
                    nodecount[V[i]][0] += 1
                    nodecount[V[i]][1] += 1
                    V_list[V[i]] = V_list.get(V[i], 0) + 1
                    newindeg = sum(list(S_in[V[i]].values())) - V_list.get(V[i], 0) * V[i][1]
                    newoutdeg = sum(list(S_out[V[i]].values())) - V_list.get(V[i], 0) * V[i][0]
                V_list[(newoutdeg, newindeg)] = V_list.get((newoutdeg, newindeg), 0) + 1
            j = 0
            while j < newindeg:
                outnode_choice = random.sample(S_in[V[i]].keys(), 1)
                outnode = outnode_choice[0]
                if S_out[outnode][V[i]] > 0:
                    # 更新JDM
                    JDM[(outnode, (newoutdeg, newindeg))] = JDM.get((outnode, (newoutdeg, newindeg)), 0) + 1
                    JDM[(outnode, V[i])] = JDM[(outnode, V[i])] - 1
                    # 更新S_in、S_out
                    S_in[(newoutdeg, newindeg)] = S_in.get((newoutdeg, newindeg), {})
                    S_in[(newoutdeg, newindeg)][outnode] = S_in[(newoutdeg, newindeg)].get(outnode, 0) + 1
                    S_in[V[i]][outnode] -= 1

                    S_out[outnode][(newoutdeg, newindeg)] = S_out[outnode].get((newoutdeg, newindeg), 0) + 1
                    S_out[outnode][V[i]] = S_out[outnode][V[i]] - 1
                    j += 1
                else:
                    continue
            j = 0
            while j < newoutdeg:
                innode_choice = random.sample(S_out[V[i]].keys(), 1)
                innode = innode_choice[0]
                if S_in[innode][V[i]] > 0:
                    # 更新JDM
                    JDM[((newoutdeg, newindeg), innode)] = JDM.get(((newoutdeg, newindeg), innode), 0) + 1
                    JDM[(V[i], innode)] = JDM[(V[i], innode)] - 1
                    # 更新S_in、S_out
                    S_out[(newoutdeg, newindeg)] = S_out.get((newoutdeg, newindeg), {})
                    S_out[(newoutdeg, newindeg)][innode] = S_out[(newoutdeg, newindeg)].get(innode, 0) + 1
                    S_out[V[i]][innode] -= 1

                    S_in[innode][(newoutdeg, newindeg)] = S_in[innode].get((newoutdeg, newindeg), 0) + 1
                    S_in[innode][V[i]] = S_in[innode][V[i]] - 1
                    j += 1
                else:
                    continue
    print(V_list)

    # 检测多边问题
    JDM_items = list(JDM.items())
    for i in range(len(JDM_items)):
        in_node = JDM_items[i][0][1]
        out_node = JDM_items[i][0][0]
        if JDM_items[i][1] == 0:
            continue
        elif in_node==out_node and JDM_items[i][1]> V_list.get(in_node, 0)*(V_list.get(in_node, 0)-1):
            JDM[(out_node, in_node)] = V_list.get(in_node, 0)*(V_list.get(in_node, 0)-1)
            out_unsat[out_node] = out_unsat.get(out_node, 0) + JDM_items[i][1] - V_list.get(in_node, 0)*(V_list.get(in_node, 0)-1)
            in_unsat[in_node] = in_unsat.get(in_node, 0) + JDM_items[i][1] - V_list.get(in_node, 0)*(V_list.get(in_node, 0)-1)
        elif JDM_items[i][1]> V_list.get(in_node, 0) * V_list.get(out_node, 0):
            JDM[(out_node, in_node)] = V_list.get(in_node, 0) * V_list.get(out_node, 0)
            out_unsat[out_node] = out_unsat.get(out_node, 0) + JDM_items[i][1] - V_list.get(in_node, 0) * V_list.get(
                out_node, 0)
            in_unsat[in_node] = in_unsat.get(in_node, 0) + JDM_items[i][1] - V_list.get(in_node, 0) * V_list.get(
                out_node, 0)
    """
        解决不饱和问题，随机连接不饱和点
        
    """
    sorted_out_unsat = sorted(out_unsat.keys(), key=lambda out_unsat: out_unsat[1] + out_unsat[0], reverse=True)
    sorted_in_unsat = sorted(in_unsat.keys(), key=lambda in_unsat: in_unsat[1] + in_unsat[0], reverse=True)

    for i in range(len(sorted_out_unsat)):
        out_node = sorted_out_unsat[i]
        if out_unsat[out_node] > 0:
            for j in range(len(sorted_in_unsat)):
                in_node = sorted_in_unsat[j]
                if in_unsat[in_node] > 0:
                    if out_node != in_node:
                        if JDM.get((out_node, in_node), 0) >= V_list[in_node] * V_list[
                            out_node]:  # nodecount[in_node[0]][1] * nodecount[out_node[0]][0]:
                            continue
                        else:
                            if in_unsat[in_node] >= out_unsat[out_node]:
                                diff = min(out_unsat[out_node],
                                           V_list[in_node] * V_list[out_node] - JDM.get((out_node, in_node), 0))
                                JDM[(out_node, in_node)] = JDM.get((out_node, in_node), 0) + diff
                                out_unsat[out_node] -= diff
                                in_unsat[in_node] -= diff
                            elif in_unsat[in_node] < out_unsat[out_node]:
                                diff = min(in_unsat[in_node],
                                           V_list[in_node] * V_list[out_node] - JDM.get((out_node, in_node), 0))
                                JDM[(out_node, in_node)] = JDM.get((out_node, in_node), 0) + diff
                                out_unsat[out_node] -= diff
                                in_unsat[in_node] -= diff
                            if out_unsat[out_node] == 0:
                                break
                    elif out_node == in_node:
                        if JDM.get((out_node, in_node), 0) >= V_list.get(in_node, 0)*(V_list.get(in_node, 0)-1):  # (nodecount[in_node[0]][1] - 1) * (nodecount[out_node[0]][0] - 1):
                            continue
                            # break
                        elif out_unsat[out_node[0]] > 0:
                            JDM[(out_node[0], in_node[0])] = JDM.get((out_node[0], in_node[0]), 0) + 1
                            out_unsat[out_node[0]] -= 1
                            in_unsat[in_node[0]] -= 1
                            if out_unsat[out_node] == 0:
                                break
    while sum(out_unsat.values()) > 0:
        out_node = random.sample(out_unsat.keys(), 1)
        if out_unsat[out_node[0]] == 0:
            del out_unsat[out_node[0]]
        else:
            JDM[(out_node[0], (0, 1))] = JDM.get((out_node[0], (0, 1)), 0) + 1
            out_unsat[out_node[0]] -= 1
            V_list[(0, 1)] = V_list.get((0, 1), 0) + 1
    while sum(in_unsat.values()) > 0:
        in_node = random.sample(in_unsat.keys(), 1)
        if in_unsat[in_node[0]] == 0:
            del in_unsat[in_node[0]]
        else:
            JDM[((1, 0), in_node[0])] = JDM.get(((1, 0), in_node[0]), 0) + 1
            in_unsat[in_node[0]] -= 1
            if in_unsat[in_node[0]] == 0:
                del in_unsat[in_node[0]]
            V_list[(1, 0)] = V_list.get((1, 0), 0) + 1

    # print("V_list", V_list)
    # sumin = 0
    # sumout = 0
    # for i in S_in.keys():
    #     sumin = sumin + sum(list(S_in[i].values()))
    # for i in S_out.keys():
    #     sumout = sumout + sum(list(S_out[i].values()))
    # print("sumin", sumin)
    # print("sumout", sumout)
    # print("JDM sum", sum(list(JDM.values())))
    #
    # sumin = 0
    # sumout = 0
    # for i in V_list:
    #     sumin = sumin + V_list[i] * i[1]
    #     sumout = sumout + V_list[i] * i[0]
    # print("sumin", sumin)
    # print("sumout", sumout)

    return V_list, JDM



# p2p-Gnutella08.txt
# Wiki-Vote.txt
# example.txt
# soc-Wiki-Vote.txt
g = nx.read_edgelist("D:\\pythonversion\\data\\p2p-Gnutella08.txt", create_using=nx.DiGraph)
# JDM=GO.getQuintuple(g)
JDM={((10, 0), (0, 1)): 91, ((10, 0), (0, 6)): 18, ((10, 0), (10, 77)): 8, ((10, 0), (10, 59)): 1, ((10, 0), (10, 73)): 5, ((10, 0), (9, 51)): 2, ((10, 0), (10, 74)): 4, ((10, 0), (10, 49)): 1, ((10, 0), (0, 7)): 15, ((10, 77), (10, 5)): 2, ((10, 77), (0, 4)): 1, ((10, 77), (0, 7)): 1, ((10, 77), (10, 7)): 1, ((10, 77), (0, 12)): 1, ((10, 77), (1, 1)): 1, ((10, 77), (0, 1)): 1, ((10, 77), (0, 3)): 2, ((10, 77), (0, 2)): 1, ((10, 59), (9, 70)): 1, ((10, 59), (0, 2)): 2, ((10, 59), (0, 11)): 3, ((10, 59), (9, 4)): 1, ((10, 59), (0, 7)): 1, ((10, 59), (0, 6)): 1, ((10, 59), (0, 1)): 1, ((10, 73), (0, 54)): 1, ((10, 73), (10, 85)): 2, ((10, 73), (0, 50)): 1, ((10, 73), (9, 71)): 1, ((10, 73), (10, 60)): 1, ((10, 73), (8, 82)): 1, ((10, 73), (10, 81)): 1, ((10, 73), (10, 70)): 1, ((10, 73), (9, 83)): 1, ((10, 73), (0, 19)): 1, ((9, 51), (9, 81)): 1, ((9, 51), (9, 66)): 1, ((9, 51), (9, 70)): 1, ((9, 51), (10, 70)): 1, ((9, 51), (9, 14)): 1, ((9, 51), (0, 52)): 1, ((9, 51), (10, 20)): 1, ((9, 51), (0, 5)): 2, ((10, 74), (0, 7)): 1, ((10, 74), (10, 7)): 1, ((10, 74), (10, 13)): 1, ((10, 74), (10, 2)): 1, ((10, 74), (9, 7)): 1, ((10, 74), (10, 11)): 1, ((10, 74), (9, 5)): 1, ((10, 74), (0, 1)): 1, ((10, 74), (0, 2)): 1, ((10, 74), (8, 1)): 1, ((10, 49), (10, 77)): 1, ((10, 49), (9, 67)): 1, ((10, 49), (9, 70)): 1, ((10, 49), (0, 21)): 1, ((10, 49), (10, 60)): 1, ((10, 49), (10, 57)): 1, ((10, 49), (8, 82)): 1, ((10, 49), (10, 38)): 1, ((10, 49), (10, 81)): 1, ((10, 49), (9, 47)): 1, ((10, 5), (2, 11)): 2, ((10, 5), (0, 4)): 61, ((10, 5), (10, 5)): 24, ((10, 5), (9, 6)): 5, ((10, 5), (0, 3)): 69, ((10, 5), (10, 2)): 36, ((10, 5), (10, 3)): 45, ((10, 5), (0, 2)): 53, ((10, 7), (9, 21)): 1, ((10, 7), (9, 3)): 3, ((10, 7), (0, 4)): 23, ((10, 7), (10, 4)): 10, ((10, 7), (10, 8)): 4, ((10, 7), (10, 2)): 12, ((10, 7), (0, 3)): 20, ((10, 7), (10, 5)): 9, ((10, 7), (0, 5)): 19, ((1, 1), (0, 5)): 9, ((10, 5), (0, 14)): 11, ((10, 5), (10, 8)): 9, ((10, 5), (0, 9)): 17, ((10, 5), (0, 5)): 33, ((9, 70), (10, 74)): 1, ((9, 70), (0, 54)): 1, ((9, 70), (10, 69)): 1, ((9, 70), (10, 72)): 1, ((9, 70), (9, 60)): 2, ((9, 70), (0, 21)): 1, ((9, 70), (8, 82)): 1, ((9, 70), (10, 81)): 1, ((9, 70), (8, 86)): 2, ((9, 4), (13, 7)): 1, ((9, 4), (10, 5)): 8, ((9, 4), (0, 3)): 21, ((9, 4), (0, 2)): 23, ((9, 4), (9, 2)): 3, ((9, 4), (9, 4)): 5, ((9, 4), (0, 5)): 12, ((10, 85), (10, 49)): 1, ((10, 85), (9, 70)): 2, ((10, 85), (10, 56)): 1, ((10, 85), (10, 60)): 1, ((10, 85), (10, 81)): 1, ((10, 85), (0, 10)): 1, ((10, 85), (10, 20)): 1, ((10, 85), (0, 22)): 1, ((10, 85), (0, 2)): 1, ((9, 71), (10, 59)): 2, ((9, 71), (10, 49)): 1, ((9, 71), (9, 30)): 1, ((9, 71), (10, 27)): 1, ((9, 71), (10, 87)): 1, ((9, 71), (10, 72)): 2, ((9, 71), (10, 38)): 1, ((9, 71), (10, 66)): 1, ((9, 71), (0, 44)): 1, ((10, 60), (0, 3)): 3, ((10, 60), (4, 7)): 1, ((10, 60), (5, 7)): 1, ((10, 60), (0, 2)): 2, ((10, 60), (1, 4)): 1, ((10, 60), (0, 1)): 2, ((8, 82), (10, 87)): 1, ((8, 82), (10, 38)): 1, ((8, 82), (10, 81)): 1, ((8, 82), (10, 15)): 1, ((8, 82), (9, 14)): 1, ((8, 82), (0, 4)): 1, ((8, 82), (10, 20)): 1, ((8, 82), (0, 11)): 1, ((10, 81), (8, 82)): 1, ((10, 81), (10, 31)): 1, ((10, 81), (9, 18)): 1, ((10, 81), (10, 6)): 1, ((10, 81), (0, 6)): 2, ((10, 81), (9, 2)): 1, ((10, 81), (10, 2)): 1, ((10, 81), (10, 5)): 2, ((10, 81), (0, 4)): 1, ((10, 81), (10, 3)): 1, ((10, 70), (10, 69)): 1, ((10, 70), (10, 87)): 1, ((10, 70), (10, 85)): 1, ((10, 70), (8, 82)): 1, ((10, 70), (10, 81)): 1, ((10, 70), (8, 86)): 1, ((10, 70), (0, 67)): 1, ((10, 70), (0, 32)): 1, ((10, 70), (0, 52)): 1, ((10, 70), (9, 60)): 1, ((9, 83), (0, 9)): 1, ((9, 83), (0, 5)): 1, ((9, 83), (0, 3)): 2, ((9, 83), (10, 6)): 1, ((9, 83), (0, 4)): 2, ((9, 83), (0, 13)): 1, ((9, 83), (0, 1)): 1, ((9, 81), (10, 78)): 1, ((9, 81), (0, 21)): 1, ((9, 81), (0, 25)): 1, ((9, 81), (8, 86)): 1, ((9, 81), (10, 69)): 1, ((9, 81), (9, 56)): 1, ((9, 81), (10, 47)): 1, ((9, 81), (10, 79)): 1, ((9, 81), (0, 10)): 1, ((9, 66), (10, 73)): 1, ((9, 66), (10, 57)): 1, ((9, 66), (9, 63)): 1, ((9, 66), (10, 31)): 1, ((9, 66), (10, 5)): 1, ((9, 66), (0, 13)): 1, ((9, 66), (0, 6)): 1, ((9, 66), (10, 35)): 1, ((9, 66), (0, 1)): 1, ((9, 70), (10, 71)): 1, ((9, 70), (9, 81)): 1, ((9, 70), (10, 78)): 1, ((9, 70), (0, 91)): 1, ((9, 70), (0, 67)): 1, ((9, 70), (10, 66)): 1, ((9, 70), (10, 62)): 1, ((9, 14), (10, 49)): 1, ((9, 14), (0, 54)): 1, ((9, 14), (10, 69)): 1, ((9, 14), (10, 85)): 1, ((9, 14), (0, 50)): 1, ((9, 14), (10, 61)): 1, ((9, 14), (9, 81)): 1, ((9, 14), (9, 70)): 1, ((9, 14), (8, 86)): 1, ((10, 20), (10, 77)): 1, ((10, 20), (10, 56)): 1, ((10, 20), (9, 66)): 1, ((10, 20), (10, 81)): 2, ((10, 20), (0, 91)): 1, ((10, 20), (10, 66)): 1, ((10, 20), (0, 52)): 1, ((10, 20), (10, 35)): 1, ((10, 20), (0, 44)): 1, ((10, 7), (10, 59)): 2, ((10, 7), (10, 73)): 6, ((10, 7), (10, 49)): 1, ((10, 7), (10, 67)): 2, ((10, 7), (9, 47)): 2, ((10, 7), (9, 71)): 2, ((10, 7), (10, 70)): 3, ((10, 7), (0, 32)): 3, ((10, 7), (9, 56)): 2, ((10, 13), (10, 38)): 1, ((10, 13), (10, 3)): 1, ((10, 13), (0, 7)): 3, ((10, 13), (10, 8)): 1, ((10, 13), (0, 3)): 6, ((10, 13), (10, 2)): 1, ((10, 13), (0, 1)): 2, ((10, 13), (10, 4)): 3, ((10, 13), (0, 5)): 3, ((10, 2), (10, 49)): 10, ((10, 2), (10, 73)): 29, ((10, 2), (10, 61)): 16, ((10, 2), (9, 70)): 31, ((10, 2), (10, 57)): 12, ((10, 2), (0, 91)): 17, ((10, 2), (10, 70)): 23, ((10, 2), (0, 52)): 13, ((10, 2), (10, 79)): 12, ((10, 2), (0, 44)): 8, ((9, 7), (10, 69)): 2, ((9, 7), (10, 72)): 2, ((9, 7), (0, 3)): 6, ((9, 7), (9, 60)): 1, ((9, 7), (10, 57)): 1, ((9, 7), (8, 82)): 1, ((9, 7), (10, 81)): 4, ((9, 7), (8, 86)): 2, ((9, 7), (10, 62)): 2, ((10, 11), (10, 3)): 2, ((10, 11), (0, 9)): 1, ((10, 11), (10, 8)): 2, ((10, 11), (5, 5)): 1, ((10, 11), (10, 2)): 3, ((10, 11), (0, 1)): 7, ((10, 11), (10, 1)): 3, ((10, 11), (0, 3)): 3, ((10, 11), (1, 2)): 1, ((9, 5), (9, 51)): 1, ((9, 5), (10, 61)): 1, ((9, 5), (9, 70)): 2, ((9, 5), (10, 78)): 3, ((9, 5), (10, 62)): 2, ((9, 5), (10, 60)): 1, ((9, 5), (10, 57)): 2, ((9, 5), (9, 47)): 3, ((9, 5), (10, 70)): 1, ((8, 1), (10, 59)): 1, ((8, 1), (9, 51)): 1, ((8, 1), (10, 61)): 2, ((8, 1), (9, 81)): 1, ((8, 1), (10, 78)): 1, ((8, 1), (10, 62)): 1, ((8, 1), (0, 91)): 1, ((8, 1), (0, 32)): 1, ((10, 77), (10, 69)): 2, ((10, 77), (10, 87)): 1, ((10, 77), (9, 81)): 1, ((10, 77), (9, 67)): 1, ((10, 77), (9, 66)): 1, ((10, 77), (10, 57)): 1, ((10, 77), (10, 20)): 1, ((10, 77), (0, 5)): 1, ((9, 67), (10, 74)): 1, ((9, 67), (0, 50)): 1, ((9, 67), (9, 66)): 1, ((9, 67), (9, 70)): 1, ((9, 67), (10, 81)): 2, ((9, 67), (8, 86)): 1, ((9, 67), (9, 83)): 1, ((9, 67), (0, 22)): 1, ((10, 57), (10, 6)): 1, ((10, 57), (1, 3)): 1, ((10, 57), (0, 1)): 3, ((10, 57), (0, 3)): 1, ((10, 57), (9, 8)): 1, ((10, 57), (9, 4)): 1, ((10, 57), (0, 8)): 1, ((10, 57), (9, 5)): 1, ((10, 38), (0, 5)): 1, ((10, 38), (10, 5)): 1, ((10, 38), (0, 4)): 1, ((10, 38), (0, 3)): 1, ((10, 38), (10, 2)): 2, ((10, 38), (10, 4)): 1, ((10, 38), (0, 7)): 1, ((10, 38), (0, 44)): 1, ((10, 38), (0, 1)): 1, ((10, 81), (0, 8)): 2, ((10, 81), (10, 16)): 1, ((10, 81), (10, 4)): 1, ((10, 81), (0, 9)): 1, ((10, 81), (10, 10)): 1, ((10, 81), (0, 1)): 3, ((10, 81), (0, 3)): 1, ((9, 47), (9, 3)): 1, ((9, 47), (0, 5)): 1, ((9, 47), (0, 2)): 2, ((9, 47), (0, 3)): 3, ((9, 47), (0, 9)): 1, ((9, 47), (0, 6)): 1, ((10, 4), (0, 1)): 124, ((10, 4), (1, 5)): 5, ((10, 4), (10, 6)): 43, ((10, 4), (0, 2)): 99, ((10, 4), (9, 30)): 8, ((10, 4), (0, 4)): 99, ((10, 4), (10, 7)): 21, ((10, 4), (10, 10)): 13, ((1, 5), (0, 2)): 2, ((1, 5), (0, 4)): 3, ((10, 6), (10, 77)): 10, ((10, 6), (10, 87)): 3, ((10, 6), (10, 61)): 3, ((10, 6), (10, 71)): 4, ((10, 6), (10, 81)): 16, ((10, 6), (8, 86)): 5, ((10, 6), (0, 19)): 3, ((10, 6), (8, 32)): 3, ((9, 30), (0, 8)): 1, ((9, 30), (0, 10)): 1, ((9, 30), (10, 4)): 1, ((9, 30), (4, 2)): 1, ((9, 30), (0, 6)): 1, ((9, 30), (0, 5)): 1, ((9, 30), (9, 3)): 1, ((9, 30), (0, 4)): 1, ((9, 30), (0, 2)): 1, ((10, 7), (0, 54)): 2, ((10, 7), (9, 67)): 1, ((10, 7), (0, 67)): 1, ((10, 7), (10, 81)): 4, ((10, 7), (10, 47)): 1, ((10, 7), (9, 60)): 2, ((10, 7), (10, 1)): 8, ((10, 7), (0, 1)): 25, ((10, 10), (2, 13)): 1, ((10, 10), (0, 5)): 3, ((10, 10), (1, 5)): 1, ((10, 10), (0, 3)): 4, ((10, 10), (0, 4)): 9, ((10, 10), (10, 3)): 6, ((10, 10), (0, 2)): 8, ((10, 10), (0, 1)): 10, ((10, 87), (10, 77)): 1, ((10, 87), (9, 51)): 1, ((10, 87), (10, 71)): 1, ((10, 87), (9, 81)): 1, ((10, 87), (9, 60)): 1, ((10, 87), (9, 67)): 1, ((10, 87), (9, 47)): 1, ((10, 87), (8, 86)): 1, ((10, 87), (10, 69)): 1, ((10, 87), (10, 55)): 1, ((10, 61), (10, 14)): 1, ((10, 61), (0, 18)): 1, ((10, 61), (0, 4)): 2, ((10, 61), (10, 6)): 1, ((10, 61), (10, 3)): 1, ((10, 61), (10, 2)): 1, ((10, 61), (0, 3)): 1, ((10, 61), (0, 2)): 2, ((10, 71), (10, 69)): 1, ((10, 71), (10, 77)): 1, ((10, 71), (0, 50)): 1, ((10, 71), (9, 60)): 1, ((10, 71), (0, 21)): 1, ((10, 71), (8, 82)): 1, ((10, 71), (8, 86)): 1, ((10, 71), (0, 19)): 1, ((10, 71), (10, 81)): 1, ((10, 71), (0, 10)): 1, ((8, 86), (10, 59)): 1, ((8, 86), (10, 73)): 1, ((8, 86), (9, 51)): 1, ((8, 86), (10, 81)): 1, ((8, 86), (0, 91)): 1, ((8, 86), (10, 66)): 1, ((8, 86), (0, 32)): 1, ((8, 86), (10, 79)): 1, ((10, 81), (10, 59)): 1, ((10, 81), (10, 77)): 1, ((10, 81), (10, 81)): 1, ((10, 81), (0, 25)): 1, ((10, 81), (10, 70)): 1, ((10, 81), (9, 63)): 1, ((10, 81), (9, 9)): 1, ((10, 81), (0, 13)): 1, ((8, 32), (10, 4)): 1, ((8, 32), (9, 6)): 1, ((8, 32), (10, 6)): 1, ((8, 32), (0, 3)): 2, ((8, 32), (0, 12)): 1, ((8, 32), (10, 1)): 1, ((8, 32), (0, 2)): 1, ((10, 4), (0, 10)): 14, ((10, 4), (5, 6)): 1, ((10, 4), (10, 3)): 49, ((10, 4), (10, 2)): 43, ((10, 4), (10, 14)): 8, ((10, 4), (10, 5)): 28, ((10, 4), (9, 2)): 12, ((4, 2), (0, 4)): 4, ((4, 2), (0, 13)): 1, ((4, 2), (0, 2)): 2, ((9, 3), (10, 73)): 6, ((9, 3), (9, 81)): 2, ((9, 3), (9, 67)): 5, ((9, 3), (9, 47)): 2, ((9, 3), (9, 56)): 4, ((9, 3), (10, 8)): 5, ((9, 3), (0, 52)): 2, ((9, 3), (0, 4)): 37, ((9, 3), (0, 2)): 45, ((10, 47), (10, 77)): 1, ((10, 47), (10, 56)): 1, ((10, 47), (0, 25)): 1, ((10, 47), (9, 63)): 1, ((10, 47), (10, 69)): 1, ((10, 47), (0, 19)): 1, ((10, 47), (10, 81)): 1, ((10, 47), (9, 23)): 1, ((10, 47), (9, 9)): 1, ((10, 47), (0, 52)): 1, ((9, 60), (10, 73)): 1, ((9, 60), (9, 51)): 2, ((9, 60), (9, 81)): 1, ((9, 60), (10, 78)): 1, ((9, 60), (10, 62)): 1, ((9, 60), (9, 71)): 1, ((9, 60), (10, 33)): 1, ((9, 60), (10, 79)): 1, ((9, 60), (0, 44)): 1, ((10, 1), (9, 5)): 32, ((10, 1), (0, 9)): 82, ((10, 1), (0, 5)): 173, ((10, 1), (10, 6)): 109, ((10, 1), (10, 1)): 110, ((10, 1), (10, 12)): 17, ((10, 1), (0, 2)): 328, ((10, 1), (10, 3)): 159, ((10, 1), (4, 9)): 3, ((10, 1), (0, 11)): 35, ((10, 2), (0, 4)): 180, ((10, 2), (10, 6)): 70, ((10, 2), (0, 7)): 105, ((10, 2), (0, 3)): 197, ((10, 2), (9, 6)): 14, ((2, 13), (0, 9)): 1, ((2, 13), (0, 3)): 1, ((10, 3), (10, 2)): 81, ((10, 3), (1, 5)): 7, ((10, 3), (0, 13)): 25, ((10, 3), (0, 4)): 166, ((10, 3), (10, 3)): 105, ((10, 3), (0, 2)): 171, ((10, 3), (0, 5)): 108, ((10, 3), (9, 4)): 22, ((10, 3), (0, 3)): 170, ((10, 3), (0, 15)): 8, ((10, 3), (10, 8)): 21, ((10, 3), (10, 4)): 85, ((10, 3), (0, 20)): 4, ((10, 3), (0, 1)): 149, ((10, 0), (0, 15)): 2, ((10, 0), (0, 2)): 40, ((10, 0), (1, 2)): 4, ((10, 0), (10, 2)): 24, ((10, 0), (3, 2)): 1, ((10, 0), (3, 6)): 3, ((10, 0), (10, 4)): 20, ((10, 0), (10, 27)): 1, ((10, 0), (9, 6)): 2, ((1, 2), (10, 15)): 1, ((10, 2), (0, 5)): 146, ((10, 2), (0, 13)): 29, ((10, 2), (9, 3)): 34, ((10, 2), (0, 2)): 221, ((10, 2), (0, 6)): 97, ((3, 2), (10, 3)): 1, ((3, 2), (0, 5)): 3, ((3, 2), (0, 4)): 2, ((3, 6), (10, 9)): 1, ((3, 6), (2, 5)): 1, ((3, 6), (10, 6)): 1, ((10, 4), (10, 77)): 11, ((10, 4), (10, 73)): 10, ((10, 4), (10, 74)): 10, ((10, 4), (0, 54)): 11, ((10, 4), (9, 81)): 5, ((10, 4), (10, 62)): 14, ((10, 4), (8, 82)): 12, ((10, 4), (9, 83)): 11, ((10, 27), (10, 77)): 1, ((10, 27), (10, 9)): 1, ((10, 27), (0, 14)): 1, ((10, 27), (0, 8)): 1, ((10, 27), (0, 6)): 1, ((10, 27), (10, 6)): 1, ((10, 27), (10, 10)): 1, ((10, 27), (0, 5)): 1, ((10, 27), (0, 3)): 1, ((10, 27), (0, 2)): 1, ((9, 6), (0, 2)): 7, ((9, 6), (10, 4)): 3, ((9, 6), (10, 10)): 2, ((9, 6), (10, 2)): 4, ((9, 6), (9, 3)): 2, ((9, 6), (9, 2)): 5, ((9, 6), (0, 5)): 12, ((10, 15), (0, 12)): 1, ((10, 15), (0, 3)): 5, ((10, 15), (10, 5)): 2, ((10, 15), (10, 7)): 1, ((10, 15), (10, 6)): 2, ((10, 15), (10, 4)): 2, ((9, 3), (9, 7)): 4, ((9, 3), (10, 3)): 23, ((9, 3), (0, 3)): 39, ((9, 3), (0, 13)): 4, ((10, 3), (0, 14)): 23, ((10, 3), (1, 4)): 16, ((10, 3), (10, 13)): 5, ((10, 9), (10, 3)): 6, ((10, 9), (8, 7)): 1, ((10, 9), (0, 2)): 13, ((10, 9), (0, 1)): 16, ((10, 9), (5, 6)): 1, ((10, 9), (10, 2)): 8, ((10, 9), (9, 2)): 2, ((2, 5), (0, 4)): 1, ((2, 5), (0, 3)): 1, ((10, 6), (0, 13)): 9, ((10, 6), (10, 2)): 32, ((10, 6), (0, 3)): 56, ((10, 6), (0, 8)): 17, ((10, 6), (0, 5)): 39, ((10, 6), (0, 6)): 29, ((10, 6), (0, 15)): 3, ((10, 6), (0, 2)): 58, ((10, 73), (9, 63)): 1, ((10, 73), (0, 3)): 1, ((10, 73), (0, 10)): 1, ((10, 73), (0, 52)): 1, ((10, 73), (0, 4)): 1, ((10, 73), (0, 13)): 1, ((10, 73), (0, 6)): 1, ((10, 73), (10, 35)): 1, ((10, 73), (0, 11)): 1, ((10, 62), (10, 77)): 1, ((10, 62), (10, 73)): 4, ((10, 62), (9, 51)): 1, ((10, 62), (10, 69)): 1, ((10, 62), (0, 3)): 1, ((10, 62), (9, 81)): 1, ((10, 62), (8, 82)): 1, ((10, 62), (10, 81)): 1, ((10, 62), (9, 56)): 1, ((10, 9), (0, 4)): 17, ((10, 9), (10, 4)): 6, ((10, 9), (0, 14)): 1, ((10, 9), (0, 3)): 9, ((10, 9), (1, 2)): 1, ((10, 6), (10, 9)): 7, ((10, 6), (0, 7)): 26, ((10, 6), (9, 11)): 1, ((10, 6), (10, 6)): 29, ((10, 6), (0, 9)): 19, ((10, 6), (9, 9)): 3, ((10, 6), (0, 4)): 51, ((10, 10), (9, 21)): 1, ((10, 10), (0, 7)): 2, ((10, 10), (10, 9)): 1, ((10, 10), (1, 4)): 2, ((10, 10), (10, 2)): 3, ((10, 4), (10, 81)): 24, ((10, 4), (0, 15)): 5, ((10, 4), (0, 7)): 41, ((10, 4), (4, 9)): 1, ((10, 4), (0, 5)): 63, ((10, 10), (10, 77)): 2, ((10, 10), (0, 54)): 1, ((10, 10), (10, 87)): 1, ((10, 10), (10, 72)): 2, ((10, 10), (10, 85)): 2, ((10, 10), (0, 50)): 1, ((10, 10), (10, 41)): 1, ((10, 10), (10, 81)): 4, ((10, 10), (10, 69)): 2, ((10, 10), (0, 19)): 1, ((10, 2), (9, 23)): 3, ((10, 2), (10, 5)): 79, ((10, 2), (10, 3)): 117, ((10, 2), (5, 5)): 8, ((10, 2), (5, 1)): 7, ((10, 2), (10, 4)): 83, ((9, 3), (0, 15)): 5, ((9, 3), (0, 9)): 15, ((9, 3), (0, 8)): 12, ((9, 3), (9, 5)): 4, ((9, 3), (0, 1)): 36, ((9, 3), (10, 2)): 20, ((9, 2), (0, 15)): 2, ((9, 2), (0, 3)): 47, ((9, 2), (0, 8)): 23, ((9, 2), (10, 6)): 20, ((9, 2), (0, 4)): 56, ((9, 2), (0, 1)): 55, ((9, 2), (0, 9)): 21, ((9, 2), (10, 2)): 20, ((9, 2), (10, 7)): 11, ((9, 2), (1, 4)): 2, ((9, 2), (10, 3)): 23, ((9, 2), (0, 2)): 37, ((9, 2), (7, 6)): 1, ((9, 2), (2, 8)): 1, ((9, 2), (10, 4)): 24, ((9, 2), (9, 2)): 5, ((9, 2), (5, 4)): 1, ((10, 2), (0, 1)): 219, ((10, 2), (9, 21)): 4, ((10, 2), (10, 2)): 92, ((10, 3), (10, 77)): 21, ((10, 3), (10, 87)): 15, ((10, 3), (10, 73)): 21, ((10, 3), (9, 81)): 15, ((10, 3), (10, 62)): 17, ((10, 3), (9, 71)): 19, ((10, 3), (10, 60)): 8, ((10, 3), (8, 82)): 16, ((10, 3), (10, 81)): 27, ((10, 3), (9, 83)): 15, ((9, 21), (0, 11)): 1, ((9, 21), (0, 9)): 1, ((9, 21), (0, 7)): 1, ((9, 21), (0, 5)): 1, ((9, 21), (10, 5)): 1, ((9, 21), (0, 3)): 2, ((9, 21), (0, 1)): 1, ((9, 21), (0, 4)): 1, ((10, 2), (10, 7)): 38, ((10, 2), (6, 1)): 2, ((10, 2), (10, 8)): 27, ((10, 2), (0, 8)): 62, ((10, 6), (0, 11)): 6, ((10, 6), (10, 3)): 31, ((10, 6), (9, 1)): 4, ((10, 6), (9, 3)): 12, ((10, 5), (2, 5)): 1, ((10, 5), (13, 9)): 1, ((10, 5), (1, 2)): 13, ((10, 5), (0, 1)): 71, ((10, 5), (0, 6)): 29, ((10, 5), (10, 4)): 31, ((10, 5), (0, 7)): 27, ((10, 5), (1, 3)): 3, ((10, 5), (10, 1)): 17, ((10, 7), (10, 16)): 1, ((10, 7), (0, 9)): 7, ((10, 7), (10, 12)): 3, ((10, 7), (1, 2)): 3, ((6, 1), (0, 5)): 2, ((6, 1), (10, 6)): 2, ((6, 1), (0, 2)): 5, ((6, 1), (0, 4)): 3, ((6, 1), (10, 2)): 1, ((10, 8), (4, 9)): 1, ((10, 8), (10, 5)): 4, ((10, 8), (0, 8)): 5, ((10, 8), (0, 4)): 9, ((10, 8), (10, 3)): 8, ((10, 8), (0, 2)): 22, ((10, 8), (0, 1)): 16, ((10, 8), (0, 5)): 7, ((10, 8), (10, 7)): 2, ((10, 5), (10, 73)): 4, ((10, 5), (0, 50)): 3, ((10, 5), (9, 81)): 3, ((10, 5), (10, 78)): 7, ((10, 5), (10, 62)): 3, ((10, 5), (9, 70)): 12, ((10, 5), (9, 71)): 8, ((10, 5), (0, 91)): 6, ((10, 5), (10, 35)): 2, ((10, 5), (0, 44)): 6, ((10, 3), (0, 9)): 41, ((10, 3), (0, 10)): 27, ((10, 3), (9, 2)): 22, ((10, 3), (9, 1)): 9, ((9, 1), (0, 12)): 9, ((9, 1), (0, 7)): 17, ((9, 1), (0, 8)): 14, ((9, 1), (0, 14)): 4, ((9, 1), (0, 4)): 49, ((9, 1), (0, 2)): 54, ((9, 1), (0, 3)): 66, ((9, 1), (1, 4)): 6, ((9, 3), (10, 59)): 6, ((9, 3), (9, 70)): 4, ((9, 3), (10, 56)): 4, ((9, 3), (8, 86)): 4, ((9, 3), (10, 69)): 5, ((9, 3), (10, 66)): 4, ((9, 3), (8, 32)): 2, ((10, 0), (0, 4)): 31, ((10, 0), (0, 3)): 36, ((10, 0), (9, 8)): 2, ((10, 0), (0, 5)): 21, ((9, 8), (10, 4)): 1, ((9, 8), (0, 7)): 1, ((9, 8), (0, 6)): 2, ((9, 8), (0, 3)): 4, ((9, 8), (0, 2)): 3, ((9, 8), (0, 4)): 3, ((9, 8), (10, 1)): 2, ((10, 4), (10, 71)): 5, ((10, 4), (9, 67)): 8, ((10, 4), (9, 63)): 8, ((10, 4), (10, 72)): 10, ((10, 4), (10, 66)): 6, ((10, 4), (8, 32)): 6, ((10, 4), (0, 44)): 4, ((10, 1), (0, 15)): 16, ((10, 1), (0, 8)): 75, ((10, 1), (0, 7)): 124, ((10, 1), (0, 1)): 386, ((10, 1), (10, 4)): 146, ((10, 2), (0, 10)): 41, ((10, 2), (0, 11)): 31, ((10, 2), (10, 1)): 54, ((10, 2), (9, 11)): 4, ((9, 3), (1, 4)): 2, ((9, 3), (0, 14)): 5, ((9, 3), (10, 4)): 17, ((10, 1), (10, 2)): 148, ((10, 1), (10, 5)): 98, ((10, 1), (0, 3)): 290, ((10, 1), (24, 3)): 1, ((10, 1), (1, 1)): 34, ((9, 3), (0, 6)): 16, ((9, 3), (0, 5)): 21, ((9, 11), (0, 4)): 2, ((9, 11), (5, 2)): 1, ((9, 11), (0, 1)): 1, ((9, 11), (0, 2)): 3, ((9, 11), (0, 9)): 1, ((9, 11), (10, 2)): 1, ((9, 11), (1, 3)): 1, ((10, 5), (3, 2)): 1, ((10, 5), (5, 12)): 1, ((10, 5), (8, 6)): 1, ((10, 5), (10, 15)): 5, ((10, 5), (5, 2)): 5, ((10, 5), (1, 1)): 11, ((10, 4), (0, 6)): 57, ((10, 4), (10, 1)): 34, ((1, 4), (0, 9)): 2, ((10, 2), (5, 4)): 3, ((10, 2), (3, 2)): 3, ((9, 5), (0, 5)): 15, ((9, 5), (0, 8)): 3, ((9, 5), (0, 4)): 16, ((9, 5), (10, 5)): 8, ((9, 5), (0, 1)): 30, ((9, 5), (10, 2)): 9, ((9, 5), (0, 2)): 13, ((9, 5), (0, 3)): 11, ((10, 3), (10, 6)): 63, ((10, 3), (9, 3)): 19, ((10, 3), (11, 3)): 1, ((10, 4), (10, 4)): 43, ((10, 4), (8, 7)): 2, ((10, 4), (5, 3)): 4, ((10, 4), (0, 3)): 90, ((10, 2), (1, 2)): 24, ((10, 2), (1, 3)): 20, ((10, 2), (0, 50)): 15, ((10, 2), (9, 81)): 19, ((10, 2), (9, 67)): 15, ((10, 2), (9, 83)): 17, ((10, 2), (22, 3)): 2, ((24, 3), (0, 7)): 2, ((24, 3), (0, 11)): 1, ((24, 3), (10, 15)): 1, ((24, 3), (0, 3)): 2, ((24, 3), (0, 12)): 1, ((24, 3), (0, 8)): 2, ((24, 3), (0, 6)): 2, ((24, 3), (10, 35)): 1, ((24, 3), (10, 5)): 1, ((24, 3), (0, 5)): 1, ((24, 3), (10, 13)): 1, ((24, 3), (9, 6)): 1, ((24, 3), (10, 4)): 1, ((24, 3), (10, 2)): 1, ((24, 3), (5, 2)): 1, ((24, 3), (0, 2)): 2, ((24, 3), (10, 9)): 1, ((24, 3), (0, 1)): 2, ((1, 1), (10, 7)): 1, ((10, 1), (9, 6)): 17, ((10, 1), (10, 11)): 12, ((10, 1), (1, 5)): 7, ((10, 1), (5, 2)): 13, ((1, 1), (0, 1)): 19, ((10, 4), (8, 3)): 3, ((10, 4), (10, 11)): 8, ((10, 4), (0, 12)): 12, ((10, 3), (0, 7)): 64, ((10, 3), (29, 4)): 1, ((10, 3), (10, 1)): 49, ((10, 3), (5, 1)): 9, ((5, 2), (0, 5)): 8, ((5, 2), (0, 8)): 7, ((5, 2), (0, 2)): 14, ((5, 2), (0, 6)): 6, ((5, 2), (0, 4)): 16, ((10, 2), (10, 10)): 22, ((1, 3), (10, 10)): 1, ((3, 2), (9, 6)): 1, ((3, 2), (10, 7)): 1, ((3, 2), (0, 6)): 1, ((5, 12), (0, 8)): 1, ((5, 12), (9, 6)): 1, ((5, 12), (0, 5)): 1, ((5, 12), (0, 3)): 1, ((5, 12), (0, 1)): 1, ((8, 6), (10, 7)): 1, ((8, 6), (10, 1)): 2, ((8, 6), (10, 4)): 1, ((8, 6), (0, 3)): 1, ((8, 6), (9, 7)): 1, ((8, 6), (0, 5)): 2, ((8, 6), (10, 3)): 4, ((8, 6), (0, 1)): 1, ((10, 15), (9, 4)): 1, ((10, 15), (9, 3)): 2, ((10, 15), (1, 3)): 1, ((10, 15), (0, 2)): 4, ((10, 15), (5, 3)): 1, ((10, 15), (0, 1)): 2, ((10, 3), (0, 54)): 5, ((10, 3), (0, 50)): 6, ((10, 3), (9, 70)): 23, ((10, 3), (10, 67)): 13, ((10, 3), (0, 25)): 7, ((10, 3), (0, 67)): 9, ((10, 3), (10, 66)): 10, ((1, 1), (0, 3)): 6, ((10, 1), (10, 7)): 41, ((10, 1), (9, 8)): 6, ((10, 1), (5, 5)): 7, ((10, 1), (0, 4)): 253, ((10, 0), (10, 5)): 15, ((10, 2), (10, 20)): 3, ((10, 2), (0, 9)): 56, ((10, 2), (0, 14)): 12, ((10, 2), (0, 18)): 2, ((10, 2), (0, 20)): 11, ((10, 5), (10, 47)): 6, ((10, 5), (9, 2)): 7, ((10, 4), (0, 9)): 28, ((10, 4), (0, 18)): 1, ((10, 4), (0, 13)): 12, ((10, 4), (9, 6)): 8, ((10, 7), (0, 7)): 18, ((10, 7), (9, 6)): 4, ((10, 7), (0, 8)): 6, ((5, 5), (10, 5)): 3, ((5, 5), (0, 10)): 1, ((5, 5), (9, 9)): 1, ((5, 5), (10, 4)): 2, ((5, 5), (0, 3)): 4, ((10, 6), (10, 41)): 3, ((10, 6), (10, 4)): 37, ((10, 6), (0, 1)): 73, ((10, 6), (3, 8)): 1, ((10, 4), (9, 5)): 11, ((10, 3), (10, 5)): 63, ((10, 3), (10, 12)): 7, ((10, 3), (0, 8)): 33, ((10, 3), (0, 6)): 77, ((10, 5), (10, 12)): 6, ((10, 5), (5, 3)): 3, ((10, 5), (9, 1)): 5, ((10, 2), (9, 2)): 29, ((10, 3), (9, 47)): 6, ((9, 2), (9, 3)): 4, ((9, 2), (0, 7)): 26, ((9, 2), (9, 9)): 3, ((9, 2), (0, 6)): 25, ((9, 2), (10, 1)): 12, ((10, 8), (10, 2)): 4, ((10, 8), (9, 7)): 1, ((10, 8), (0, 3)): 15, ((10, 8), (1, 1)): 2, ((10, 8), (9, 1)): 1, ((10, 6), (9, 21)): 2, ((10, 6), (0, 91)): 7, ((10, 6), (10, 10)): 6, ((10, 6), (10, 5)): 25, ((9, 6), (0, 9)): 5, ((9, 6), (18, 7)): 1, ((9, 6), (0, 8)): 5, ((9, 6), (10, 5)): 6, ((9, 6), (10, 1)): 2, ((9, 6), (0, 3)): 10, ((9, 6), (0, 1)): 14, ((10, 6), (7, 6)): 1, ((10, 6), (1, 2)): 6, ((7, 6), (0, 11)): 2, ((7, 6), (2, 11)): 1, ((7, 6), (0, 8)): 2, ((7, 6), (0, 5)): 1, ((7, 6), (0, 2)): 4, ((7, 6), (0, 4)): 4, ((7, 6), (9, 2)): 1, ((1, 2), (10, 1)): 2, ((10, 4), (4, 7)): 2, ((10, 4), (5, 2)): 1, ((10, 4), (1, 1)): 11, ((10, 4), (1, 3)): 11, ((10, 4), (5, 5)): 2, ((10, 6), (9, 51)): 3, ((10, 6), (10, 74)): 5, ((10, 6), (10, 49)): 1, ((10, 6), (10, 67)): 3, ((10, 6), (10, 15)): 2, ((10, 6), (10, 20)): 1, ((10, 6), (10, 35)): 4, ((2, 11), (10, 5)): 1, ((2, 11), (0, 6)): 1, ((9, 2), (0, 12)): 9, ((9, 2), (0, 5)): 27, ((9, 2), (0, 14)): 5, ((10, 1), (10, 10)): 20, ((10, 1), (3, 8)): 2, ((4, 7), (0, 5)): 1, ((4, 7), (0, 3)): 2, ((4, 7), (0, 1)): 3, ((5, 2), (0, 15)): 1, ((5, 2), (9, 5)): 1, ((1, 1), (0, 7)): 4, ((1, 3), (0, 4)): 6, ((5, 5), (1, 4)): 3, ((5, 5), (10, 8)): 1, ((5, 5), (0, 13)): 2, ((5, 5), (10, 1)): 1, ((10, 67), (0, 54)): 1, ((10, 67), (10, 69)): 1, ((10, 67), (10, 87)): 1, ((10, 67), (10, 77)): 1, ((10, 67), (9, 66)): 1, ((10, 67), (0, 21)): 1, ((10, 67), (10, 57)): 1, ((10, 67), (8, 86)): 1, ((10, 67), (10, 72)): 1, ((10, 67), (10, 62)): 1, ((10, 15), (0, 54)): 1, ((10, 15), (9, 66)): 1, ((10, 15), (9, 71)): 1, ((10, 15), (10, 70)): 1, ((10, 15), (10, 3)): 1, ((10, 15), (10, 2)): 1, ((10, 35), (10, 6)): 1, ((10, 35), (0, 10)): 1, ((10, 35), (10, 14)): 2, ((10, 35), (0, 4)): 1, ((10, 35), (0, 3)): 1, ((10, 35), (10, 2)): 2, ((10, 35), (0, 5)): 1, ((10, 35), (1, 6)): 1, ((10, 1), (3, 2)): 1, ((10, 1), (2, 13)): 3, ((10, 1), (0, 12)): 27, ((10, 4), (10, 8)): 18, ((10, 5), (10, 7)): 9, ((10, 5), (0, 8)): 17, ((10, 5), (10, 10)): 6, ((10, 8), (10, 4)): 9, ((10, 8), (0, 12)): 2, ((10, 8), (5, 1)): 1, ((9, 6), (10, 6)): 6, ((9, 6), (0, 6)): 6, ((10, 0), (1, 3)): 3, ((10, 0), (5, 12)): 1, ((10, 0), (10, 3)): 16, ((10, 0), (0, 9)): 10, ((10, 0), (10, 9)): 5, ((1, 3), (10, 5)): 1, ((1, 2), (0, 1)): 5, ((10, 3), (10, 7)): 39, ((10, 3), (0, 11)): 19, ((10, 3), (1, 2)): 17, ((10, 9), (1, 3)): 2, ((10, 9), (0, 5)): 8, ((10, 9), (10, 5)): 5, ((10, 9), (10, 1)): 4, ((10, 5), (0, 12)): 8, ((10, 5), (9, 4)): 12, ((9, 63), (10, 49)): 1, ((9, 63), (9, 81)): 1, ((9, 63), (10, 62)): 1, ((9, 63), (0, 21)): 1, ((9, 63), (10, 81)): 1, ((9, 63), (0, 91)): 1, ((9, 63), (10, 72)): 1, ((9, 63), (10, 66)): 1, ((9, 63), (0, 32)): 1, ((10, 72), (0, 2)): 2, ((10, 72), (0, 8)): 1, ((10, 72), (0, 3)): 3, ((10, 72), (9, 2)): 1, ((10, 72), (0, 7)): 1, ((10, 72), (5, 3)): 1, ((10, 72), (10, 7)): 1, ((10, 72), (0, 1)): 4, ((10, 66), (0, 4)): 1, ((10, 66), (10, 8)): 1, ((10, 66), (10, 4)): 1, ((10, 66), (0, 5)): 1, ((10, 66), (0, 2)): 1, ((10, 66), (10, 7)): 1, ((10, 66), (10, 3)): 1, ((10, 66), (0, 7)): 1, ((10, 66), (10, 12)): 1, ((10, 66), (0, 6)): 1, ((10, 7), (10, 77)): 1, ((10, 7), (9, 51)): 3, ((10, 7), (10, 72)): 3, ((10, 7), (0, 50)): 1, ((10, 7), (9, 81)): 4, ((10, 7), (9, 63)): 1, ((10, 7), (9, 83)): 1, ((10, 7), (10, 79)): 2, ((10, 2), (2, 3)): 4, ((10, 2), (9, 1)): 15, ((10, 4), (9, 70)): 12, ((10, 4), (10, 78)): 4, ((10, 4), (9, 66)): 5, ((10, 4), (10, 60)): 7, ((10, 4), (9, 71)): 15, ((1, 2), (0, 7)): 3, ((10, 6), (9, 63)): 2, ((10, 5), (3, 6)): 2, ((10, 5), (0, 10)): 8, ((10, 5), (8, 3)): 4, ((10, 5), (0, 13)): 6, ((1, 3), (0, 6)): 1, ((10, 3), (9, 66)): 13, ((10, 3), (10, 41)): 5, ((10, 3), (9, 63)): 13, ((10, 3), (0, 19)): 1, ((10, 3), (10, 79)): 15, ((10, 5), (10, 69)): 6, ((10, 5), (10, 72)): 7, ((10, 5), (10, 55)): 3, ((10, 5), (10, 66)): 3, ((10, 5), (10, 11)): 5, ((10, 5), (0, 52)): 2, ((10, 5), (9, 60)): 7, ((10, 1), (10, 72)): 25, ((10, 1), (9, 2)): 43, ((10, 0), (10, 8)): 6, ((10, 0), (9, 7)): 4, ((10, 0), (10, 11)): 5, ((10, 8), (0, 14)): 1, ((10, 8), (0, 6)): 6, ((10, 4), (10, 9)): 12, ((10, 4), (9, 18)): 1, ((10, 4), (1, 2)): 10, ((10, 4), (7, 6)): 3, ((9, 7), (0, 9)): 4, ((9, 7), (0, 5)): 5, ((9, 7), (0, 2)): 9, ((9, 7), (0, 1)): 4, ((9, 7), (10, 3)): 6, ((9, 7), (10, 2)): 3, ((9, 7), (10, 4)): 2, ((10, 11), (9, 30)): 1, ((10, 11), (10, 71)): 1, ((10, 11), (9, 70)): 1, ((10, 11), (9, 81)): 1, ((10, 11), (9, 71)): 1, ((10, 11), (0, 67)): 1, ((10, 11), (9, 83)): 1, ((10, 11), (0, 52)): 1, ((10, 11), (10, 79)): 2, ((10, 11), (0, 44)): 1, ((10, 5), (7, 7)): 1, ((10, 4), (9, 3)): 14, ((10, 4), (3, 9)): 1, ((10, 5), (10, 77)): 9, ((10, 5), (0, 54)): 3, ((10, 5), (8, 82)): 5, ((10, 5), (10, 81)): 13, ((10, 5), (10, 85)): 5, ((10, 5), (10, 61)): 3, ((10, 5), (9, 66)): 4, ((10, 5), (10, 41)): 5, ((10, 5), (10, 70)): 2, ((10, 9), (10, 11)): 1, ((10, 9), (9, 5)): 1, ((10, 9), (0, 7)): 3, ((10, 9), (10, 6)): 5, ((10, 9), (0, 6)): 5, ((9, 18), (0, 4)): 1, ((9, 18), (0, 9)): 1, ((9, 18), (0, 8)): 2, ((9, 18), (0, 3)): 3, ((9, 18), (10, 2)): 1, ((9, 18), (0, 1)): 1, ((1, 2), (0, 5)): 5, ((10, 4), (10, 59)): 4, ((10, 4), (10, 67)): 6, ((10, 4), (9, 60)): 8, ((7, 6), (10, 6)): 1, ((7, 6), (10, 4)): 1, ((10, 8), (0, 7)): 11, ((10, 8), (9, 5)): 3, ((10, 8), (5, 4)): 1, ((10, 3), (9, 51)): 8, ((10, 3), (10, 49)): 6, ((10, 3), (10, 69)): 18, ((10, 3), (10, 85)): 15, ((10, 3), (10, 57)): 11, ((10, 3), (10, 70)): 8, ((10, 2), (10, 78)): 16, ((10, 2), (10, 56)): 9, ((10, 2), (10, 33)): 6, ((10, 2), (0, 22)): 5, ((10, 3), (28, 2)): 1, ((10, 3), (10, 9)): 14, ((10, 4), (4, 3)): 2, ((10, 4), (2, 6)): 3, ((10, 4), (9, 4)): 4, ((10, 4), (9, 9)): 3, ((10, 79), (0, 7)): 1, ((10, 79), (19, 4)): 1, ((10, 79), (10, 4)): 1, ((10, 79), (0, 14)): 1, ((10, 79), (9, 2)): 1, ((10, 79), (0, 2)): 1, ((10, 79), (0, 3)): 1, ((10, 79), (0, 4)): 1, ((10, 79), (0, 1)): 2, ((7, 7), (0, 12)): 1, ((7, 7), (1, 3)): 1, ((7, 7), (0, 9)): 1, ((7, 7), (0, 14)): 2, ((7, 7), (10, 4)): 1, ((7, 7), (0, 2)): 1, ((9, 2), (9, 51)): 2, ((9, 2), (10, 71)): 6, ((9, 2), (9, 81)): 5, ((9, 2), (8, 86)): 4, ((9, 2), (10, 69)): 9, ((9, 2), (10, 31)): 7, ((9, 2), (9, 60)): 4, ((9, 2), (9, 4)): 5, ((9, 2), (10, 62)): 6, ((10, 4), (9, 47)): 4, ((10, 4), (0, 67)): 6, ((10, 4), (10, 69)): 15, ((10, 4), (10, 55)): 4, ((10, 5), (10, 27)): 3, ((10, 2), (10, 85)): 16, ((10, 2), (9, 66)): 12, ((10, 2), (10, 60)): 16, ((10, 2), (9, 63)): 10, ((10, 2), (10, 77)): 43, ((10, 2), (0, 54)): 11, ((10, 2), (10, 69)): 24, ((10, 2), (10, 87)): 19, ((10, 2), (10, 72)): 28, ((10, 69), (0, 3)): 4, ((10, 69), (0, 4)): 3, ((10, 69), (0, 2)): 3, ((10, 69), (0, 9)): 1, ((10, 69), (0, 5)): 1, ((10, 69), (1, 5)): 1, ((10, 72), (10, 11)): 1, ((10, 72), (0, 10)): 1, ((10, 72), (0, 20)): 1, ((10, 72), (10, 3)): 2, ((10, 72), (10, 1)): 1, ((1, 5), (10, 3)): 1, ((9, 60), (10, 6)): 1, ((9, 60), (10, 3)): 2, ((9, 60), (1, 2)): 2, ((9, 60), (10, 2)): 2, ((9, 60), (0, 1)): 1, ((10, 69), (10, 9)): 1, ((10, 69), (0, 10)): 1, ((10, 69), (10, 8)): 1, ((10, 69), (10, 2)): 2, ((10, 69), (0, 1)): 2, ((10, 55), (10, 59)): 1, ((10, 55), (10, 85)): 1, ((10, 55), (10, 69)): 1, ((10, 55), (10, 81)): 1, ((10, 55), (0, 6)): 1, ((10, 55), (0, 1)): 2, ((10, 55), (0, 2)): 1, ((10, 55), (10, 3)): 1, ((10, 55), (0, 3)): 1, ((10, 11), (10, 73)): 1, ((10, 11), (10, 49)): 1, ((10, 11), (9, 67)): 1, ((10, 11), (10, 38)): 1, ((10, 11), (10, 70)): 1, ((10, 11), (10, 10)): 3, ((10, 11), (9, 14)): 2, ((10, 11), (9, 60)): 1, ((10, 11), (0, 11)): 1, ((10, 3), (0, 12)): 23, ((10, 3), (0, 18)): 4, ((10, 3), (9, 6)): 14, ((10, 3), (9, 7)): 8, ((10, 56), (0, 11)): 1, ((10, 56), (10, 3)): 2, ((10, 56), (5, 5)): 1, ((10, 56), (11, 3)): 1, ((10, 56), (10, 12)): 1, ((10, 56), (0, 2)): 4, ((10, 14), (10, 8)): 1, ((10, 14), (10, 6)): 2, ((10, 14), (7, 4)): 1, ((10, 14), (10, 7)): 2, ((10, 14), (10, 5)): 1, ((10, 14), (10, 2)): 3, ((10, 14), (0, 3)): 3, ((10, 14), (10, 10)): 1, ((10, 14), (10, 3)): 2, ((10, 6), (0, 12)): 10, ((10, 2), (8, 3)): 5, ((10, 2), (9, 5)): 22, ((10, 2), (1, 1)): 25, ((10, 3), (10, 11)): 10, ((10, 3), (10, 14)): 5, ((10, 5), (10, 6)): 26, ((10, 5), (1, 4)): 5, ((10, 5), (24, 3)): 2, ((10, 2), (10, 11)): 14, ((9, 4), (0, 11)): 3, ((9, 4), (0, 6)): 10, ((9, 4), (10, 13)): 2, ((9, 4), (0, 4)): 28, ((9, 4), (0, 7)): 13, ((9, 4), (8, 2)): 1, ((9, 4), (10, 3)): 10, ((10, 6), (10, 11)): 2, ((10, 6), (10, 1)): 15, ((1, 4), (10, 2)): 2, ((10, 1), (10, 14)): 10, ((10, 1), (0, 91)): 14, ((10, 1), (0, 32)): 7, ((10, 1), (9, 11)): 5, ((10, 1), (8, 4)): 3, ((10, 3), (10, 10)): 15, ((10, 3), (5, 5)): 4, ((1, 2), (0, 2)): 12, ((10, 2), (9, 51)): 11, ((10, 2), (10, 38)): 4, ((10, 2), (10, 9)): 18, ((10, 2), (9, 14)): 2, ((10, 10), (4, 7)): 1, ((10, 10), (1, 3)): 1, ((10, 10), (9, 3)): 3, ((10, 10), (10, 6)): 2, ((10, 10), (10, 4)): 2, ((10, 6), (0, 18)): 1, ((10, 6), (0, 10)): 9, ((7, 4), (10, 73)): 1, ((7, 4), (10, 38)): 1, ((7, 4), (0, 25)): 1, ((7, 4), (0, 3)): 2, ((7, 4), (0, 1)): 3, ((10, 7), (0, 2)): 24, ((10, 7), (19, 4)): 1, ((10, 7), (10, 3)): 14, ((10, 7), (1, 3)): 3, ((10, 7), (0, 10)): 6, ((10, 2), (10, 74)): 16, ((10, 2), (10, 41)): 8, ((10, 2), (9, 60)): 21, ((10, 10), (0, 12)): 2, ((10, 10), (10, 1)): 2, ((10, 3), (5, 2)): 4, ((10, 2), (2, 11)): 3, ((10, 2), (4, 12)): 3, ((8, 2), (10, 4)): 1, ((8, 2), (10, 9)): 3, ((8, 2), (10, 31)): 1, ((8, 2), (10, 6)): 2, ((8, 2), (0, 3)): 4, ((8, 2), (9, 1)): 2, ((8, 2), (0, 1)): 5, ((10, 2), (10, 15)): 12, ((10, 2), (5, 3)): 5, ((10, 4), (10, 12)): 7, ((10, 4), (2, 3)): 2, ((10, 4), (0, 11)): 9, ((10, 6), (9, 70)): 5, ((10, 6), (9, 81)): 1, ((10, 6), (9, 60)): 11, ((10, 6), (9, 67)): 2, ((10, 6), (10, 78)): 2, ((10, 78), (10, 73)): 1, ((10, 78), (0, 7)): 1, ((10, 78), (0, 50)): 1, ((10, 78), (10, 67)): 1, ((10, 78), (10, 41)): 1, ((10, 78), (10, 38)): 1, ((10, 78), (9, 47)): 1, ((10, 78), (0, 25)): 1, ((10, 78), (10, 70)): 1, ((10, 78), (10, 79)): 1, ((9, 56), (0, 14)): 1, ((9, 56), (0, 12)): 1, ((9, 56), (0, 4)): 1, ((9, 56), (0, 2)): 1, ((9, 56), (0, 20)): 1, ((9, 56), (10, 1)): 1, ((9, 56), (0, 9)): 1, ((9, 56), (0, 1)): 1, ((9, 56), (10, 3)): 1, ((10, 6), (9, 7)): 4, ((10, 6), (18, 7)): 2, ((10, 6), (0, 14)): 1, ((10, 3), (47, 3)): 1, ((10, 2), (1, 5)): 5, ((10, 3), (10, 59)): 7, ((10, 3), (10, 71)): 8, ((10, 3), (10, 33)): 5, ((10, 3), (0, 91)): 12, ((10, 3), (0, 52)): 5, ((1, 2), (0, 6)): 2, ((10, 2), (0, 15)): 7, ((10, 62), (9, 60)): 1, ((10, 62), (9, 67)): 1, ((10, 62), (10, 78)): 1, ((10, 62), (10, 56)): 1, ((10, 62), (10, 33)): 1, ((10, 62), (0, 67)): 1, ((10, 62), (10, 72)): 1, ((10, 62), (0, 44)): 1, ((10, 41), (10, 49)): 1, ((10, 41), (10, 71)): 1, ((10, 41), (9, 60)): 1, ((10, 41), (10, 62)): 1, ((10, 41), (9, 56)): 1, ((10, 41), (10, 3)): 1, ((10, 41), (0, 1)): 3, ((10, 41), (5, 3)): 1, ((10, 3), (8, 13)): 5, ((9, 3), (0, 54)): 1, ((9, 3), (0, 50)): 2, ((9, 3), (10, 61)): 1, ((9, 3), (10, 41)): 1, ((9, 3), (10, 70)): 2, ((9, 3), (10, 79)): 4, ((8, 13), (10, 4)): 2, ((8, 13), (0, 15)): 1, ((8, 13), (0, 12)): 1, ((8, 13), (0, 1)): 2, ((8, 13), (0, 2)): 4, ((10, 1), (10, 9)): 30, ((10, 4), (0, 8)): 25, ((10, 9), (10, 12)): 1, ((10, 9), (1, 5)): 1, ((10, 9), (0, 9)): 3, ((10, 9), (0, 8)): 3, ((10, 5), (10, 87)): 3, ((10, 6), (5, 12)): 2, ((10, 6), (9, 10)): 1, ((10, 4), (10, 61)): 2, ((10, 4), (10, 56)): 4, ((10, 4), (10, 41)): 3, ((10, 4), (0, 91)): 7, ((10, 4), (10, 70)): 4, ((10, 3), (10, 61)): 9, ((10, 3), (0, 22)): 2, ((9, 10), (1, 2)): 1, ((9, 10), (10, 4)): 1, ((9, 10), (10, 6)): 2, ((9, 10), (10, 3)): 1, ((9, 10), (10, 5)): 1, ((9, 10), (2, 4)): 1, ((9, 10), (9, 2)): 1, ((9, 10), (0, 3)): 1, ((10, 4), (8, 6)): 1, ((10, 1), (9, 1)): 26, ((10, 1), (10, 8)): 41, ((10, 1), (1, 2)): 29, ((10, 3), (10, 56)): 5, ((10, 7), (0, 15)): 2, ((10, 7), (9, 23)): 2, ((10, 7), (10, 15)): 1, ((10, 7), (0, 6)): 19, ((10, 7), (0, 12)): 2, ((10, 33), (10, 77)): 1, ((10, 33), (0, 3)): 3, ((10, 33), (0, 9)): 1, ((10, 33), (4, 12)): 1, ((10, 33), (0, 2)): 1, ((10, 33), (0, 1)): 1, ((10, 33), (10, 1)): 1, ((10, 33), (10, 2)): 1, ((5, 5), (5, 5)): 1, ((5, 5), (10, 79)): 1, ((5, 5), (0, 5)): 2, ((5, 5), (0, 1)): 3, ((5, 5), (10, 2)): 3, ((11, 3), (10, 5)): 2, ((11, 3), (0, 3)): 1, ((11, 3), (0, 6)): 1, ((11, 3), (0, 2)): 1, ((11, 3), (10, 3)): 1, ((11, 3), (0, 7)): 1, ((11, 3), (0, 4)): 1, ((11, 3), (10, 1)): 1, ((11, 3), (0, 1)): 2, ((10, 12), (10, 74)): 1, ((10, 12), (10, 87)): 1, ((10, 12), (10, 61)): 1, ((10, 12), (10, 62)): 2, ((10, 12), (10, 57)): 1, ((10, 12), (9, 71)): 1, ((10, 12), (9, 63)): 1, ((10, 12), (9, 83)): 1, ((10, 12), (10, 79)): 1, ((10, 12), (10, 19)): 1, ((10, 3), (1, 1)): 5, ((10, 31), (0, 14)): 1, ((10, 31), (9, 5)): 1, ((10, 31), (10, 2)): 2, ((10, 31), (0, 3)): 2, ((10, 31), (0, 9)): 1, ((10, 31), (0, 1)): 3, ((10, 31), (9, 3)): 1, ((10, 31), (0, 2)): 1, ((10, 5), (3, 11)): 4, ((10, 5), (22, 3)): 1, ((9, 23), (10, 5)): 1, ((9, 23), (10, 6)): 1, ((9, 23), (0, 4)): 1, ((9, 23), (0, 5)): 2, ((9, 23), (10, 4)): 1, ((9, 23), (0, 2)): 1, ((9, 23), (0, 1)): 1, ((9, 23), (9, 3)): 1, ((4, 12), (0, 8)): 1, ((4, 12), (0, 2)): 1, ((4, 12), (0, 3)): 1, ((4, 12), (0, 6)): 1, ((10, 1), (5, 3)): 7, ((10, 5), (0, 11)): 5, ((5, 0), (9, 21)): 1, ((5, 0), (6, 4)): 1, ((5, 0), (10, 1)): 3, ((5, 0), (9, 2)): 1, ((5, 0), (10, 4)): 1, ((6, 4), (0, 3)): 4, ((6, 4), (0, 9)): 1, ((6, 4), (10, 1)): 1, ((6, 4), (0, 1)): 3, ((6, 4), (10, 5)): 1, ((9, 2), (10, 59)): 5, ((9, 2), (10, 77)): 3, ((9, 2), (9, 66)): 4, ((9, 2), (0, 67)): 5, ((9, 2), (10, 72)): 8, ((9, 2), (10, 11)): 4, ((10, 4), (0, 21)): 3, ((10, 1), (0, 10)): 47, ((10, 1), (9, 4)): 29, ((10, 1), (1, 4)): 14, ((10, 5), (10, 9)): 4, ((10, 5), (9, 3)): 9, ((10, 31), (9, 51)): 1, ((10, 31), (9, 30)): 1, ((10, 31), (10, 69)): 1, ((10, 31), (9, 70)): 1, ((10, 31), (9, 60)): 1, ((10, 31), (10, 38)): 1, ((10, 31), (10, 70)): 1, ((10, 31), (0, 10)): 1, ((10, 11), (0, 5)): 5, ((10, 11), (8, 1)): 1, ((10, 11), (0, 7)): 1, ((10, 11), (5, 2)): 1, ((10, 6), (9, 56)): 5, ((10, 6), (5, 1)): 1, ((10, 2), (5, 7)): 1, ((10, 2), (8, 13)): 3, ((10, 1), (0, 13)): 31, ((9, 3), (10, 49)): 2, ((9, 3), (0, 7)): 18, ((9, 3), (0, 91)): 3, ((9, 3), (0, 67)): 3, ((9, 3), (10, 55)): 1, ((9, 3), (10, 62)): 4, ((10, 1), (0, 6)): 122, ((10, 1), (2, 5)): 1, ((10, 4), (0, 14)): 9, ((2, 5), (0, 1)): 1, ((2, 5), (0, 2)): 1, ((10, 11), (9, 3)): 1, ((10, 11), (10, 6)): 2, ((10, 11), (0, 4)): 4, ((10, 11), (0, 6)): 2, ((10, 11), (0, 2)): 2, ((10, 8), (10, 87)): 1, ((10, 8), (5, 3)): 2, ((10, 8), (10, 1)): 3, ((10, 6), (10, 8)): 3, ((10, 2), (5, 2)): 10, ((9, 4), (8, 3)): 2, ((9, 4), (0, 1)): 19, ((9, 4), (0, 8)): 7, ((8, 3), (0, 5)): 5, ((8, 3), (0, 11)): 2, ((8, 3), (0, 14)): 3, ((8, 3), (0, 2)): 6, ((8, 3), (0, 8)): 3, ((10, 7), (4, 20)): 1, ((10, 7), (10, 14)): 1, ((10, 7), (9, 18)): 1, ((10, 10), (10, 49)): 1, ((10, 10), (9, 81)): 1, ((10, 10), (10, 78)): 2, ((10, 10), (10, 62)): 3, ((10, 10), (9, 71)): 1, ((10, 10), (10, 79)): 2, ((9, 2), (0, 18)): 1, ((9, 2), (0, 10)): 12, ((10, 12), (10, 3)): 3, ((10, 12), (10, 4)): 1, ((10, 12), (0, 2)): 5, ((10, 12), (0, 3)): 2, ((10, 12), (10, 2)): 3, ((10, 12), (0, 4)): 5, ((10, 12), (0, 1)): 6, ((10, 12), (10, 1)): 2, ((10, 12), (0, 5)): 3, ((9, 1), (9, 30)): 5, ((9, 1), (10, 27)): 3, ((9, 1), (10, 87)): 5, ((9, 1), (10, 73)): 8, ((9, 1), (9, 81)): 2, ((9, 1), (10, 78)): 5, ((9, 1), (9, 66)): 3, ((9, 1), (10, 31)): 7, ((9, 1), (10, 79)): 6, ((10, 4), (10, 57)): 3, ((10, 5), (10, 60)): 7, ((10, 5), (8, 86)): 6, ((1, 1), (0, 2)): 19, ((1, 2), (10, 5)): 1, ((1, 1), (10, 4)): 2, ((9, 5), (10, 77)): 2, ((9, 5), (10, 56)): 3, ((9, 5), (0, 91)): 3, ((9, 5), (0, 67)): 3, ((9, 5), (10, 69)): 1, ((9, 5), (10, 55)): 1, ((9, 5), (10, 81)): 4, ((9, 5), (10, 31)): 1, ((9, 5), (9, 66)): 1, ((9, 5), (10, 7)): 5, ((9, 5), (1, 1)): 2, ((1, 1), (10, 3)): 6, ((10, 3), (10, 78)): 11, ((9, 6), (0, 14)): 3, ((9, 6), (0, 18)): 1, ((9, 6), (10, 7)): 2, ((10, 0), (10, 41)): 1, ((10, 0), (4, 6)): 1, ((4, 6), (0, 4)): 1, ((4, 6), (0, 5)): 1, ((4, 6), (0, 2)): 1, ((4, 6), (0, 3)): 1, ((10, 3), (9, 11)): 4, ((5, 3), (0, 1)): 5, ((5, 3), (0, 3)): 11, ((5, 3), (10, 3)): 5, ((5, 3), (0, 2)): 11, ((10, 6), (10, 73)): 3, ((10, 6), (9, 66)): 3, ((10, 6), (10, 57)): 3, ((10, 2), (10, 67)): 10, ((10, 2), (10, 62)): 26, ((10, 2), (0, 67)): 12, ((10, 2), (10, 55)): 14, ((10, 2), (10, 81)): 54, ((10, 2), (10, 66)): 14, ((10, 4), (13, 9)): 1, ((10, 4), (4, 2)): 2, ((5, 7), (0, 12)): 1, ((5, 7), (0, 6)): 1, ((5, 7), (0, 1)): 1, ((5, 7), (0, 7)): 1, ((5, 7), (10, 5)): 1, ((1, 4), (0, 4)): 1, ((10, 6), (9, 5)): 9, ((10, 6), (1, 1)): 2, ((9, 8), (0, 8)): 1, ((9, 8), (0, 10)): 1, ((9, 8), (0, 9)): 2, ((9, 8), (10, 3)): 2, ((9, 4), (0, 54)): 1, ((9, 4), (10, 69)): 3, ((9, 4), (10, 87)): 1, ((9, 4), (10, 73)): 2, ((9, 4), (9, 81)): 3, ((9, 4), (10, 78)): 1, ((9, 4), (10, 33)): 1, ((9, 4), (10, 38)): 1, ((9, 4), (10, 66)): 3, ((10, 2), (10, 71)): 16, ((10, 4), (9, 1)): 5, ((10, 16), (10, 74)): 1, ((10, 16), (10, 73)): 1, ((10, 16), (0, 50)): 1, ((10, 16), (10, 78)): 1, ((10, 16), (10, 56)): 1, ((10, 16), (10, 60)): 1, ((10, 16), (10, 57)): 1, ((10, 16), (0, 32)): 1, ((10, 16), (10, 47)): 1, ((10, 16), (0, 52)): 1, ((10, 4), (4, 1)): 1, ((10, 9), (10, 9)): 2, ((10, 9), (9, 3)): 3, ((10, 9), (10, 10)): 3, ((10, 9), (4, 3)): 1, ((9, 3), (10, 6)): 15, ((9, 3), (10, 5)): 13, ((9, 3), (10, 1)): 9, ((10, 5), (10, 59)): 3, ((10, 5), (9, 67)): 2, ((10, 5), (0, 25)): 1, ((9, 2), (10, 85)): 5, ((9, 2), (10, 56)): 6, ((9, 2), (8, 82)): 2, ((9, 2), (0, 91)): 7, ((9, 2), (10, 55)): 6, ((10, 2), (4, 4)): 3, ((10, 5), (2, 8)): 1, ((10, 3), (8, 9)): 2, ((9, 3), (8, 5)): 3, ((8, 5), (0, 4)): 1, ((8, 5), (0, 7)): 3, ((8, 5), (0, 3)): 4, ((8, 5), (0, 8)): 1, ((8, 5), (0, 2)): 3, ((8, 5), (10, 6)): 1, ((8, 5), (0, 5)): 2, ((10, 6), (10, 7)): 8, ((10, 6), (9, 4)): 7, ((10, 1), (0, 20)): 9, ((10, 1), (9, 3)): 45, ((9, 4), (2, 13)): 1, ((9, 4), (8, 6)): 1, ((9, 4), (0, 9)): 5, ((9, 4), (10, 4)): 7, ((9, 4), (1, 1)): 2, ((1, 2), (0, 10)): 1, ((10, 5), (9, 7)): 3, ((10, 5), (10, 13)): 4, ((10, 4), (10, 13)): 1, ((9, 7), (10, 85)): 1, ((9, 7), (10, 71)): 2, ((9, 7), (9, 47)): 3, ((9, 7), (9, 63)): 2, ((9, 7), (9, 14)): 1, ((18, 7), (0, 10)): 1, ((18, 7), (0, 7)): 1, ((18, 7), (0, 3)): 4, ((18, 7), (10, 6)): 2, ((18, 7), (10, 12)): 1, ((18, 7), (0, 6)): 2, ((18, 7), (0, 9)): 1, ((18, 7), (10, 2)): 1, ((18, 7), (0, 5)): 3, ((18, 7), (0, 2)): 2, ((10, 1), (10, 49)): 8, ((10, 1), (10, 87)): 15, ((10, 1), (10, 73)): 24, ((10, 1), (10, 61)): 11, ((10, 1), (9, 70)): 20, ((10, 1), (9, 66)): 8, ((10, 1), (9, 71)): 33, ((10, 1), (8, 86)): 12, ((10, 1), (10, 81)): 32, ((10, 4), (2, 11)): 2, ((10, 4), (8, 2)): 2, ((10, 9), (9, 7)): 1, ((9, 7), (9, 51)): 1, ((9, 7), (10, 74)): 1, ((9, 7), (0, 54)): 1, ((9, 7), (9, 70)): 1, ((9, 7), (0, 91)): 1, ((9, 7), (9, 71)): 1, ((9, 7), (9, 83)): 1, ((9, 7), (0, 32)): 1, ((10, 1), (10, 56)): 5, ((10, 1), (13, 7)): 2, ((10, 1), (0, 14)): 15, ((9, 71), (9, 60)): 1, ((9, 71), (9, 66)): 1, ((9, 71), (8, 82)): 1, ((9, 71), (0, 91)): 1, ((9, 71), (8, 86)): 1, ((9, 71), (0, 67)): 1, ((9, 71), (10, 35)): 1, ((10, 3), (9, 56)): 9, ((10, 3), (10, 47)): 5, ((13, 7), (0, 6)): 1, ((13, 7), (10, 6)): 1, ((13, 7), (0, 11)): 1, ((13, 7), (10, 5)): 1, ((13, 7), (0, 9)): 2, ((13, 7), (0, 4)): 2, ((13, 7), (9, 6)): 1, ((13, 7), (0, 5)): 1, ((13, 7), (0, 7)): 2, ((13, 7), (10, 2)): 1, ((10, 2), (9, 71)): 29, ((10, 2), (9, 56)): 12, ((10, 2), (10, 47)): 16, ((10, 1), (2, 3)): 2, ((2, 3), (0, 6)): 1, ((2, 3), (0, 4)): 2, ((10, 7), (10, 6)): 8, ((10, 7), (0, 11)): 3, ((10, 6), (5, 2)): 3, ((9, 6), (10, 77)): 1, ((9, 6), (10, 85)): 1, ((9, 6), (10, 56)): 2, ((9, 6), (10, 33)): 1, ((9, 6), (10, 81)): 2, ((9, 6), (8, 86)): 1, ((9, 6), (0, 67)): 1, ((9, 6), (10, 69)): 1, ((9, 6), (8, 32)): 1, ((9, 3), (10, 11)): 3, ((9, 3), (0, 10)): 7, ((10, 5), (2, 6)): 2, ((10, 5), (5, 4)): 2, ((10, 2), (0, 12)): 22, ((10, 3), (8, 86)): 10, ((5, 2), (0, 13)): 3, ((5, 2), (0, 11)): 5, ((5, 2), (0, 3)): 8, ((5, 2), (10, 6)): 3, ((10, 0), (1, 1)): 4, ((1, 1), (10, 2)): 5, ((10, 2), (9, 4)): 19, ((10, 2), (10, 12)): 8, ((10, 12), (9, 4)): 2, ((10, 12), (10, 7)): 1, ((10, 12), (0, 8)): 1, ((10, 7), (10, 9)): 2, ((10, 7), (1, 1)): 2, ((10, 7), (5, 5)): 2, ((10, 2), (1, 4)): 7, ((10, 6), (25, 5)): 1, ((9, 3), (1, 3)): 3, ((9, 3), (9, 3)): 3, ((9, 3), (0, 12)): 2, ((9, 3), (10, 10)): 3, ((10, 4), (1, 7)): 1, ((10, 4), (34, 9)): 2, ((10, 4), (2, 2)): 1, ((10, 10), (0, 9)): 1, ((10, 10), (10, 5)): 4, ((10, 10), (9, 1)): 1, ((10, 10), (10, 14)): 1, ((10, 2), (8, 86)): 13, ((1, 7), (0, 1)): 1, ((10, 1), (10, 16)): 2, ((10, 1), (9, 7)): 15, ((10, 4), (10, 85)): 7, ((10, 4), (10, 31)): 7, ((10, 4), (0, 32)): 2, ((10, 4), (9, 56)): 3, ((10, 4), (0, 52)): 2, ((1, 4), (10, 6)): 1, ((9, 5), (10, 73)): 6, ((9, 5), (9, 71)): 2, ((9, 5), (9, 83)): 1, ((9, 5), (0, 52)): 2, ((9, 5), (10, 35)): 2, ((9, 1), (9, 5)): 6, ((9, 1), (10, 6)): 22, ((9, 1), (0, 13)): 2, ((9, 1), (1, 2)): 11, ((9, 1), (0, 1)): 60, ((9, 1), (0, 6)): 19, ((9, 1), (10, 3)): 19, ((9, 1), (9, 4)): 2, ((10, 6), (10, 31)): 4, ((10, 6), (9, 2)): 4, ((10, 1), (0, 54)): 7, ((10, 1), (0, 50)): 4, ((10, 1), (9, 81)): 8, ((10, 1), (10, 66)): 8, ((10, 1), (10, 79)): 8, ((3, 8), (0, 5)): 1, ((3, 8), (0, 6)): 2, ((3, 8), (0, 3)): 1, ((10, 6), (10, 27)): 2, ((10, 6), (10, 66)): 3, ((9, 4), (9, 51)): 1, ((9, 4), (9, 67)): 2, ((9, 4), (10, 67)): 1, ((9, 4), (8, 86)): 3, ((9, 4), (0, 67)): 2, ((9, 4), (10, 72)): 2, ((9, 4), (10, 79)): 1, ((10, 7), (9, 7)): 1, ((10, 7), (9, 4)): 3, ((10, 3), (8, 6)): 2, ((9, 3), (1, 5)): 2, ((9, 3), (6, 4)): 1, ((9, 5), (7, 4)): 1, ((9, 5), (0, 15)): 2, ((9, 5), (5, 3)): 1, ((9, 5), (10, 3)): 6, ((10, 6), (8, 82)): 4, ((9, 4), (0, 10)): 5, ((9, 4), (10, 2)): 10, ((9, 5), (10, 15)): 1, ((9, 5), (9, 23)): 3, ((9, 5), (0, 10)): 3, ((9, 5), (10, 9)): 4, ((9, 5), (0, 7)): 10, ((9, 2), (10, 9)): 5, ((10, 3), (9, 8)): 3, ((10, 3), (9, 5)): 15, ((10, 6), (2, 3)): 2, ((10, 2), (9, 12)): 1, ((9, 5), (5, 2)): 2, ((9, 5), (0, 12)): 1, ((9, 5), (2, 8)): 1, ((9, 5), (5, 1)): 1, ((10, 4), (8, 86)): 10, ((10, 4), (10, 79)): 8, ((10, 4), (10, 47)): 4, ((10, 5), (9, 12)): 2, ((9, 9), (10, 7)): 1, ((9, 9), (10, 4)): 5, ((9, 9), (0, 9)): 2, ((9, 9), (10, 9)): 1, ((9, 9), (10, 5)): 2, ((9, 9), (0, 3)): 3, ((10, 6), (10, 62)): 4, ((9, 4), (9, 6)): 2, ((9, 4), (10, 1)): 4, ((9, 4), (3, 2)): 1, ((2, 3), (0, 11)): 1, ((2, 3), (0, 2)): 1, ((9, 4), (8, 13)): 2, ((10, 2), (9, 7)): 12, ((10, 2), (9, 9)): 8, ((9, 12), (9, 2)): 1, ((9, 12), (0, 7)): 1, ((9, 12), (2, 11)): 1, ((9, 12), (0, 6)): 1, ((9, 12), (0, 8)): 1, ((9, 12), (10, 3)): 2, ((9, 12), (0, 4)): 1, ((9, 12), (5, 3)): 1, ((6, 2), (10, 4)): 1, ((6, 2), (0, 2)): 1, ((6, 2), (0, 1)): 1, ((6, 2), (0, 13)): 1, ((6, 2), (0, 12)): 1, ((6, 2), (0, 5)): 1, ((10, 0), (9, 67)): 1, ((10, 0), (9, 70)): 4, ((10, 0), (10, 81)): 4, ((10, 0), (0, 91)): 1, ((10, 0), (8, 86)): 1, ((10, 0), (0, 67)): 1, ((10, 0), (9, 63)): 3, ((10, 2), (10, 59)): 4, ((10, 2), (0, 25)): 5, ((9, 2), (10, 61)): 2, ((9, 2), (9, 63)): 2, ((9, 2), (8, 32)): 2, ((9, 2), (0, 44)): 3, ((9, 2), (10, 19)): 2, ((9, 5), (9, 2)): 1, ((9, 5), (8, 3)): 1, ((9, 5), (0, 6)): 7, ((10, 1), (1, 3)): 24, ((10, 1), (5, 1)): 8, ((10, 19), (0, 7)): 1, ((10, 19), (0, 9)): 1, ((10, 19), (0, 3)): 1, ((10, 19), (0, 11)): 1, ((10, 19), (0, 15)): 1, ((10, 19), (0, 6)): 1, ((10, 19), (0, 5)): 1, ((10, 19), (0, 4)): 2, ((10, 19), (9, 3)): 1, ((8, 3), (10, 16)): 1, ((8, 3), (0, 6)): 4, ((8, 3), (0, 7)): 3, ((8, 3), (10, 7)): 1, ((8, 3), (10, 6)): 2, ((8, 3), (10, 2)): 2, ((10, 2), (10, 13)): 5, ((10, 2), (18, 3)): 1, ((10, 7), (10, 87)): 1, ((10, 7), (10, 71)): 2, ((10, 7), (9, 66)): 1, ((10, 7), (10, 69)): 1, ((10, 7), (10, 55)): 1, ((10, 7), (8, 32)): 1, ((10, 7), (10, 19)): 1, ((10, 6), (8, 6)): 1, ((10, 3), (3, 5)): 1, ((10, 3), (4, 3)): 4, ((8, 6), (10, 35)): 1, ((8, 6), (0, 7)): 1, ((8, 6), (0, 13)): 2, ((8, 6), (0, 2)): 1, ((8, 6), (9, 4)): 1, ((10, 4), (8, 9)): 2, ((3, 5), (0, 11)): 1, ((3, 5), (7, 6)): 1, ((3, 5), (10, 3)): 1, ((10, 6), (1, 3)): 6, ((9, 6), (9, 21)): 1, ((9, 6), (9, 23)): 2, ((9, 6), (3, 11)): 1, ((9, 6), (10, 11)): 2, ((9, 6), (9, 5)): 2, ((1, 5), (0, 10)): 1, ((10, 4), (10, 33)): 4, ((10, 4), (10, 38)): 1, ((10, 3), (4, 12)): 1, ((1, 4), (10, 15)): 1, ((10, 9), (10, 62)): 1, ((10, 9), (5, 2)): 2, ((5, 5), (10, 12)): 1, ((5, 5), (0, 7)): 1, ((5, 5), (0, 2)): 3, ((4, 3), (10, 4)): 3, ((4, 3), (0, 1)): 3, ((4, 3), (10, 3)): 1, ((4, 3), (0, 3)): 3, ((9, 4), (9, 21)): 1, ((9, 4), (10, 9)): 4, ((9, 4), (1, 2)): 5, ((10, 10), (1, 1)): 1, ((1, 3), (1, 1)): 2, ((10, 8), (10, 6)): 6, ((10, 8), (12, 8)): 1, ((10, 8), (10, 13)): 1, ((10, 8), (8, 1)): 1, ((8, 9), (0, 8)): 2, ((8, 9), (10, 5)): 1, ((8, 9), (0, 2)): 2, ((8, 9), (0, 7)): 1, ((8, 9), (0, 9)): 1, ((8, 9), (0, 1)): 1, ((10, 8), (10, 69)): 3, ((10, 8), (10, 77)): 2, ((10, 8), (9, 60)): 1, ((10, 8), (10, 57)): 1, ((10, 8), (10, 70)): 2, ((10, 8), (10, 72)): 2, ((10, 8), (10, 81)): 2, ((10, 8), (10, 47)): 1, ((10, 8), (8, 32)): 2, ((10, 8), (10, 19)): 2, ((10, 3), (6, 4)): 2, ((9, 6), (0, 4)): 8, ((9, 6), (0, 20)): 2, ((9, 6), (0, 12)): 4, ((10, 2), (8, 82)): 10, ((10, 4), (6, 4)): 1, ((9, 4), (10, 6)): 4, ((10, 13), (0, 8)): 1, ((10, 13), (3, 3)): 1, ((10, 13), (8, 13)): 1, ((10, 13), (10, 5)): 2, ((10, 0), (1, 5)): 1, ((10, 0), (4, 3)): 1, ((1, 5), (10, 4)): 1, ((4, 3), (10, 11)): 1, ((4, 3), (10, 7)): 2, ((4, 3), (10, 12)): 1, ((10, 3), (9, 60)): 12, ((10, 3), (9, 67)): 5, ((10, 3), (10, 72)): 18, ((10, 3), (0, 44)): 5, ((10, 7), (10, 11)): 1, ((10, 4), (0, 20)): 2, ((10, 4), (14, 11)): 2, ((10, 4), (1, 9)): 1, ((10, 0), (1, 4)): 2, ((10, 0), (9, 3)): 2, ((10, 0), (10, 6)): 13, ((10, 0), (8, 5)): 1, ((1, 4), (0, 1)): 1, ((9, 3), (10, 81)): 12, ((9, 3), (0, 11)): 5, ((8, 5), (10, 12)): 1, ((8, 5), (0, 1)): 2, ((8, 5), (9, 2)): 2, ((10, 8), (0, 9)): 3, ((10, 8), (1, 2)): 2, ((10, 6), (10, 14)): 2, ((10, 3), (4, 20)): 5, ((9, 3), (10, 77)): 4, ((9, 3), (9, 30)): 1, ((9, 3), (9, 66)): 3, ((9, 3), (10, 72)): 7, ((9, 3), (0, 32)): 3, ((9, 3), (9, 60)): 5, ((10, 12), (10, 77)): 1, ((10, 12), (10, 69)): 1, ((10, 12), (10, 12)): 1, ((10, 12), (9, 6)): 1, ((10, 12), (10, 5)): 1, ((10, 12), (9, 7)): 1, ((9, 2), (0, 11)): 6, ((9, 2), (10, 5)): 16, ((10, 0), (10, 69)): 4, ((10, 0), (10, 87)): 1, ((10, 0), (10, 72)): 4, ((10, 0), (9, 71)): 2, ((10, 0), (10, 55)): 1, ((10, 0), (9, 83)): 2, ((9, 2), (1, 2)): 5, ((5, 3), (0, 4)): 5, ((5, 3), (10, 9)): 1, ((10, 7), (9, 5)): 5, ((10, 7), (5, 3)): 2, ((10, 7), (1, 4)): 1, ((10, 3), (10, 35)): 5, ((10, 6), (0, 21)): 1, ((10, 6), (0, 20)): 1, ((10, 2), (0, 19)): 3, ((9, 9), (0, 8)): 3, ((9, 9), (0, 2)): 2, ((9, 9), (0, 20)): 2, ((9, 9), (0, 11)): 1, ((9, 9), (9, 7)): 1, ((9, 9), (0, 6)): 1, ((9, 9), (10, 2)): 2, ((10, 0), (10, 1)): 9, ((10, 0), (5, 3)): 1, ((10, 0), (1, 6)): 1, ((10, 0), (3, 5)): 1, ((10, 1), (8, 3)): 8, ((5, 3), (10, 7)): 1, ((5, 3), (9, 3)): 1, ((5, 3), (0, 5)): 7, ((5, 3), (0, 6)): 5, ((1, 6), (10, 85)): 1, ((3, 5), (9, 6)): 1, ((3, 5), (0, 4)): 1, ((3, 5), (10, 9)): 1, ((10, 6), (9, 30)): 1, ((10, 6), (5, 5)): 1, ((10, 5), (0, 20)): 1, ((8, 3), (10, 74)): 1, ((8, 3), (9, 11)): 1, ((8, 3), (1, 3)): 1, ((8, 3), (0, 20)): 1, ((8, 3), (10, 3)): 3, ((8, 3), (0, 3)): 7, ((10, 6), (9, 6)): 1, ((10, 2), (7, 6)): 3, ((10, 9), (13, 7)): 1, ((10, 9), (9, 6)): 1, ((10, 9), (1, 1)): 2, ((10, 3), (2, 2)): 2, ((10, 0), (10, 7)): 5, ((10, 0), (9, 2)): 3, ((10, 7), (10, 10)): 3, ((10, 7), (10, 7)): 2, ((10, 4), (0, 50)): 7, ((9, 2), (10, 73)): 6, ((9, 2), (9, 67)): 5, ((9, 2), (10, 41)): 1, ((9, 2), (0, 22)): 1, ((10, 10), (9, 60)): 1, ((10, 10), (10, 67)): 1, ((10, 10), (9, 47)): 1, ((10, 10), (8, 86)): 1, ((10, 10), (8, 32)): 1, ((9, 4), (0, 15)): 3, ((9, 4), (0, 12)): 6, ((9, 4), (10, 10)): 1, ((10, 2), (3, 1)): 1, ((4, 2), (10, 7)): 1, ((4, 2), (10, 2)): 1, ((4, 2), (0, 1)): 1, ((9, 2), (9, 21)): 1, ((9, 2), (2, 5)): 3, ((9, 2), (5, 5)): 1, ((5, 5), (10, 3)): 2, ((5, 3), (10, 6)): 2, ((10, 6), (4, 20)): 1, ((10, 8), (10, 85)): 4, ((10, 8), (9, 70)): 1, ((10, 8), (0, 11)): 1, ((10, 8), (10, 60)): 1, ((10, 8), (8, 82)): 1, ((10, 8), (8, 86)): 1, ((10, 2), (1, 6)): 5, ((5, 3), (3, 9)): 1, ((5, 3), (0, 11)): 1, ((10, 3), (1, 3)): 10, ((10, 9), (1, 4)): 2, ((9, 3), (1, 2)): 3, ((9, 3), (8, 9)): 1, ((1, 4), (0, 6)): 1, ((10, 3), (2, 11)): 2, ((10, 4), (10, 87)): 7, ((9, 5), (10, 72)): 3, ((9, 5), (0, 50)): 1, ((9, 5), (9, 81)): 2, ((9, 5), (10, 33)): 2, ((9, 5), (10, 38)): 1, ((9, 5), (9, 60)): 2, ((1, 1), (10, 16)): 1, ((1, 3), (0, 12)): 1, ((10, 1), (9, 60)): 20, ((10, 1), (4, 20)): 6, ((10, 3), (7, 3)): 2, ((9, 5), (10, 85)): 1, ((9, 5), (10, 71)): 3, ((9, 5), (8, 82)): 1, ((9, 5), (9, 4)): 2, ((1, 2), (0, 3)): 10, ((10, 5), (1, 5)): 1, ((10, 8), (9, 2)): 2, ((10, 8), (0, 13)): 4, ((2, 11), (0, 1)): 1, ((2, 11), (0, 5)): 1, ((10, 6), (10, 13)): 2, ((10, 5), (8, 5)): 1, ((10, 5), (10, 20)): 1, ((10, 5), (8, 9)): 2, ((10, 6), (10, 56)): 3, ((10, 6), (10, 33)): 4, ((10, 6), (10, 60)): 4, ((4, 20), (0, 7)): 1, ((4, 20), (0, 1)): 1, ((4, 20), (0, 2)): 1, ((4, 20), (10, 5)): 1, ((10, 14), (10, 73)): 1, ((10, 14), (9, 51)): 1, ((10, 14), (10, 69)): 1, ((10, 14), (10, 87)): 1, ((10, 14), (10, 78)): 1, ((10, 14), (9, 66)): 1, ((10, 14), (9, 71)): 1, ((10, 14), (8, 82)): 1, ((10, 14), (10, 81)): 1, ((10, 14), (10, 47)): 1, ((9, 6), (0, 13)): 4, ((9, 6), (9, 4)): 1, ((10, 1), (3, 3)): 2, ((3, 3), (0, 4)): 1, ((3, 3), (0, 2)): 1, ((3, 3), (10, 8)): 1, ((10, 5), (10, 67)): 4, ((10, 5), (0, 67)): 2, ((10, 5), (10, 79)): 6, ((10, 2), (6, 4)): 2, ((10, 4), (1, 6)): 2, ((10, 2), (3, 11)): 3, ((10, 1), (10, 77)): 19, ((1, 6), (0, 8)): 1, ((10, 8), (10, 11)): 1, ((10, 8), (5, 7)): 1, ((10, 8), (10, 12)): 2, ((10, 8), (1, 3)): 1, ((2, 1), (0, 10)): 1, ((2, 1), (13, 9)): 1, ((13, 9), (0, 4)): 3, ((13, 9), (0, 5)): 3, ((13, 9), (0, 9)): 1, ((13, 9), (0, 3)): 2, ((13, 9), (0, 6)): 3, ((13, 9), (0, 2)): 1, ((10, 3), (9, 30)): 2, ((10, 3), (4, 6)): 1, ((10, 3), (9, 10)): 2, ((10, 3), (14, 11)): 3, ((2, 4), (9, 5)): 1, ((2, 4), (0, 1)): 1, ((9, 2), (10, 27)): 2, ((9, 2), (9, 71)): 4, ((9, 2), (10, 33)): 1, ((9, 2), (10, 38)): 3, ((9, 2), (9, 83)): 2, ((10, 3), (5, 4)): 5, ((10, 3), (5, 3)): 4, ((10, 1), (10, 67)): 13, ((10, 1), (10, 78)): 12, ((10, 1), (10, 57)): 8, ((10, 0), (3, 8)): 1, ((3, 8), (0, 9)): 1, ((3, 8), (0, 1)): 1, ((5, 1), (0, 15)): 4, ((5, 1), (10, 14)): 1, ((5, 1), (0, 3)): 19, ((5, 1), (10, 6)): 9, ((5, 1), (10, 4)): 6, ((10, 1), (1, 9)): 2, ((10, 6), (10, 16)): 2, ((10, 6), (4, 9)): 1, ((10, 6), (10, 59)): 4, ((10, 6), (9, 23)): 1, ((4, 9), (0, 2)): 1, ((4, 9), (0, 4)): 1, ((4, 9), (0, 1)): 2, ((10, 5), (1, 9)): 1, ((10, 5), (5, 5)): 4, ((10, 4), (3, 6)): 1, ((9, 3), (10, 7)): 12, ((10, 2), (9, 47)): 3, ((10, 2), (10, 31)): 10, ((10, 8), (9, 10)): 1, ((10, 8), (10, 20)): 1, ((10, 3), (1, 6)): 5, ((10, 12), (4, 7)): 1, ((10, 12), (0, 13)): 1, ((10, 12), (1, 4)): 1, ((10, 12), (5, 5)): 1, ((10, 3), (4, 7)): 3, ((4, 7), (0, 11)): 1, ((4, 7), (0, 7)): 1, ((14, 11), (0, 4)): 2, ((14, 11), (10, 1)): 1, ((14, 11), (0, 2)): 2, ((14, 11), (0, 1)): 1, ((14, 11), (0, 5)): 1, ((14, 11), (0, 9)): 1, ((14, 11), (10, 3)): 1, ((14, 11), (0, 15)): 1, ((14, 11), (0, 10)): 1, ((14, 11), (0, 7)): 1, ((14, 11), (10, 4)): 1, ((14, 11), (0, 3)): 1, ((10, 4), (10, 16)): 2, ((3, 11), (0, 5)): 2, ((3, 11), (10, 16)): 1, ((10, 12), (0, 9)): 1, ((10, 12), (10, 20)): 1, ((10, 12), (10, 6)): 1, ((10, 12), (0, 12)): 1, ((10, 12), (0, 7)): 1, ((10, 3), (8, 1)): 1, ((9, 5), (10, 12)): 1, ((9, 5), (1, 4)): 1, ((9, 5), (10, 1)): 3, ((10, 4), (4, 8)): 1, ((9, 4), (29, 4)): 1, ((9, 4), (0, 13)): 4, ((10, 1), (5, 4)): 10, ((10, 2), (4, 1)): 2, ((10, 5), (9, 5)): 8, ((1, 3), (10, 13)): 1, ((10, 4), (5, 7)): 1, ((10, 4), (8, 13)): 2, ((10, 4), (5, 4)): 2, ((1, 3), (0, 10)): 2, ((10, 1), (3, 5)): 4, ((10, 1), (1, 6)): 4, ((10, 8), (10, 74)): 1, ((10, 8), (0, 54)): 1, ((10, 8), (9, 67)): 1, ((10, 8), (9, 66)): 1, ((10, 8), (10, 41)): 1, ((10, 8), (9, 71)): 3, ((10, 8), (9, 83)): 1, ((9, 4), (9, 1)): 2, ((9, 2), (1, 1)): 2, ((10, 2), (10, 16)): 2, ((9, 2), (9, 12)): 2, ((9, 2), (9, 5)): 3, ((5, 2), (10, 8)): 1, ((10, 2), (3, 8)): 3, ((10, 2), (9, 18)): 2, ((10, 10), (10, 8)): 3, ((10, 10), (10, 7)): 2, ((10, 10), (0, 11)): 2, ((9, 2), (10, 74)): 4, ((9, 2), (10, 8)): 6, ((9, 2), (10, 10)): 2, ((10, 5), (10, 74)): 4, ((10, 1), (10, 33)): 1, ((10, 1), (0, 52)): 8, ((10, 1), (10, 62)): 19, ((10, 1), (0, 44)): 6, ((1, 4), (10, 9)): 1, ((10, 3), (2, 13)): 3, ((10, 3), (1, 9)): 1, ((9, 3), (1, 1)): 3, ((10, 6), (10, 72)): 6, ((10, 6), (10, 38)): 1, ((10, 6), (10, 55)): 2, ((10, 6), (0, 52)): 2, ((5, 1), (0, 7)): 11, ((5, 1), (10, 2)): 5, ((10, 1), (8, 82)): 10, ((10, 1), (10, 70)): 13, ((10, 1), (4, 3)): 3, ((9, 0), (10, 2)): 6, ((9, 0), (9, 5)): 1, ((9, 0), (10, 9)): 2, ((9, 0), (0, 2)): 19, ((9, 0), (0, 1)): 17, ((9, 0), (0, 3)): 7, ((10, 4), (10, 15)): 2, ((1, 2), (3, 8)): 1, ((1, 0), (5, 3)): 1, ((5, 3), (0, 8)): 1, ((5, 3), (0, 14)): 2, ((9, 2), (10, 49)): 2, ((9, 2), (9, 47)): 2, ((9, 2), (0, 25)): 1, ((9, 2), (0, 19)): 1, ((10, 5), (10, 56)): 2, ((10, 5), (9, 83)): 1, ((10, 8), (1, 6)): 2, ((10, 8), (0, 22)): 1, ((5, 5), (1, 5)): 1, ((9, 4), (9, 3)): 2, ((9, 3), (10, 31)): 2, ((10, 10), (9, 8)): 1, ((10, 10), (9, 4)): 1, ((10, 10), (0, 10)): 1, ((10, 10), (7, 7)): 1, ((10, 10), (9, 6)): 2, ((10, 1), (3, 9)): 2, ((5, 1), (10, 7)): 4, ((5, 1), (1, 1)): 2, ((5, 1), (9, 3)): 2, ((10, 3), (2, 8)): 3, ((10, 3), (7, 2)): 1, ((10, 5), (13, 7)): 1, ((10, 5), (4, 1)): 1, ((4, 1), (0, 11)): 1, ((4, 1), (0, 9)): 2, ((4, 1), (10, 6)): 2, ((4, 1), (0, 1)): 3, ((9, 1), (10, 4)): 17, ((9, 1), (0, 9)): 14, ((9, 1), (10, 2)): 19, ((9, 1), (10, 5)): 24, ((5, 3), (0, 10)): 1, ((5, 3), (10, 10)): 1, ((9, 7), (10, 6)): 2, ((9, 7), (0, 8)): 1, ((9, 3), (9, 11)): 1, ((9, 3), (10, 14)): 4, ((9, 3), (8, 2)): 2, ((5, 3), (10, 4)): 1, ((5, 3), (0, 22)): 1, ((1, 1), (0, 20)): 1, ((10, 13), (0, 4)): 3, ((10, 13), (10, 1)): 1, ((22, 3), (9, 6)): 2, ((22, 3), (10, 3)): 2, ((22, 3), (10, 7)): 1, ((22, 3), (0, 9)): 1, ((22, 3), (0, 4)): 1, ((22, 3), (0, 44)): 1, ((22, 3), (1, 2)): 1, ((22, 3), (10, 6)): 3, ((22, 3), (0, 6)): 1, ((22, 3), (0, 10)): 1, ((22, 3), (10, 5)): 1, ((22, 3), (0, 1)): 2, ((22, 3), (10, 4)): 1, ((22, 3), (0, 2)): 1, ((22, 3), (9, 3)): 1, ((22, 3), (10, 2)): 1, ((22, 3), (10, 1)): 1, ((10, 4), (2, 5)): 1, ((8, 3), (0, 4)): 6, ((8, 3), (9, 3)): 3, ((3, 9), (0, 3)): 1, ((3, 9), (0, 4)): 1, ((3, 9), (0, 1)): 1, ((10, 7), (2, 5)): 1, ((10, 7), (2, 2)): 1, ((9, 6), (10, 31)): 1, ((9, 6), (5, 2)): 1, ((10, 5), (18, 7)): 1, ((10, 3), (13, 9)): 2, ((10, 3), (8, 2)): 3, ((10, 6), (0, 67)): 4, ((10, 6), (10, 69)): 3, ((10, 6), (10, 79)): 2, ((10, 5), (10, 57)): 3, ((10, 5), (0, 32)): 2, ((10, 5), (9, 56)): 4, ((9, 1), (0, 10)): 7, ((9, 1), (9, 6)): 5, ((9, 3), (10, 67)): 3, ((9, 3), (10, 78)): 1, ((1, 2), (10, 3)): 2, ((2, 2), (0, 15)): 1, ((2, 2), (0, 1)): 2, ((8, 13), (4, 4)): 1, ((8, 13), (0, 7)): 1, ((8, 13), (0, 10)): 1, ((8, 13), (0, 4)): 1, ((8, 13), (5, 1)): 1, ((8, 13), (0, 5)): 1, ((10, 2), (8, 32)): 9, ((10, 1), (10, 71)): 13, ((10, 1), (9, 67)): 11, ((10, 1), (9, 47)): 11, ((10, 1), (10, 69)): 19, ((10, 1), (10, 55)): 9, ((10, 1), (10, 31)): 12, ((9, 0), (0, 4)): 9, ((9, 0), (1, 2)): 3, ((9, 0), (9, 2)): 1, ((9, 0), (0, 11)): 3, ((9, 0), (10, 5)): 3, ((9, 0), (1, 3)): 2, ((10, 2), (10, 35)): 3, ((1, 3), (0, 14)): 1, ((10, 11), (0, 8)): 1, ((10, 3), (7, 6)): 2, ((10, 3), (10, 20)): 5, ((10, 3), (10, 16)): 1, ((10, 3), (7, 7)): 1, ((1, 4), (0, 7)): 1, ((10, 1), (0, 22)): 5, ((10, 1), (10, 35)): 4, ((1, 2), (0, 9)): 2, ((10, 7), (4, 9)): 1, ((1, 6), (0, 6)): 1, ((8, 1), (2, 11)): 1, ((8, 1), (0, 10)): 3, ((8, 1), (0, 5)): 5, ((8, 1), (8, 1)): 1, ((8, 1), (1, 2)): 1, ((8, 1), (0, 4)): 4, ((8, 1), (1, 4)): 1, ((8, 1), (10, 7)): 3, ((5, 2), (10, 3)): 4, ((5, 2), (0, 1)): 8, ((5, 2), (10, 5)): 3, ((10, 2), (10, 27)): 5, ((8, 1), (10, 5)): 1, ((8, 1), (0, 3)): 12, ((8, 1), (10, 10)): 1, ((8, 1), (10, 6)): 5, ((8, 1), (10, 2)): 4, ((8, 1), (10, 1)): 3, ((1, 4), (0, 2)): 1, ((10, 8), (10, 67)): 1, ((10, 8), (10, 79)): 1, ((10, 6), (2, 13)): 1, ((9, 5), (10, 10)): 1, ((9, 5), (14, 3)): 1, ((10, 6), (8, 2)): 1, ((10, 6), (17, 3)): 1, ((9, 1), (0, 5)): 35, ((10, 4), (3, 2)): 1, ((10, 7), (10, 62)): 1, ((10, 7), (0, 25)): 1, ((10, 7), (10, 66)): 1, ((10, 3), (10, 74)): 7, ((10, 3), (0, 21)): 1, ((1, 2), (9, 11)): 1, ((10, 0), (10, 33)): 1, ((10, 0), (10, 60)): 1, ((10, 0), (10, 38)): 1, ((10, 0), (10, 15)): 2, ((10, 0), (0, 19)): 1, ((10, 0), (9, 23)): 1, ((10, 0), (9, 14)): 1, ((10, 0), (0, 52)): 1, ((10, 4), (9, 21)): 1, ((10, 9), (10, 77)): 1, ((10, 9), (10, 7)): 1, ((10, 9), (5, 5)): 1, ((8, 2), (0, 9)): 2, ((8, 2), (0, 4)): 5, ((8, 2), (9, 4)): 1, ((8, 2), (0, 7)): 3, ((8, 2), (10, 8)): 1, ((10, 2), (9, 30)): 5, ((10, 1), (9, 30)): 2, ((1, 3), (10, 4)): 1, ((10, 1), (10, 15)): 4, ((1, 2), (0, 12)): 2, ((10, 9), (0, 12)): 1, ((10, 9), (2, 3)): 1, ((10, 3), (3, 8)): 2, ((1, 0), (0, 3)): 1, ((10, 2), (8, 5)): 3, ((10, 0), (0, 10)): 6, ((9, 11), (10, 59)): 1, ((9, 11), (10, 87)): 1, ((9, 11), (10, 85)): 1, ((9, 11), (0, 67)): 1, ((9, 11), (10, 69)): 1, ((9, 11), (10, 62)): 1, ((9, 11), (0, 5)): 1, ((9, 11), (10, 1)): 1, ((10, 14), (0, 9)): 2, ((10, 14), (0, 5)): 2, ((10, 14), (0, 2)): 2, ((10, 14), (0, 13)): 1, ((10, 14), (8, 3)): 1, ((10, 2), (10, 19)): 4, ((10, 1), (10, 47)): 10, ((10, 5), (9, 63)): 1, ((10, 7), (10, 57)): 1, ((10, 7), (0, 19)): 1, ((10, 7), (9, 14)): 1, ((10, 3), (10, 31)): 4, ((9, 3), (10, 47)): 1, ((9, 3), (9, 18)): 3, ((9, 3), (2, 13)): 1, ((9, 3), (17, 2)): 1, ((10, 4), (9, 7)): 2, ((1, 4), (10, 7)): 1, ((10, 3), (10, 27)): 2, ((10, 3), (0, 32)): 2, ((10, 4), (2, 8)): 1, ((10, 4), (4, 20)): 2, ((1, 4), (10, 4)): 2, ((10, 4), (9, 51)): 2, ((10, 5), (9, 21)): 3, ((10, 6), (0, 54)): 2, ((10, 6), (0, 50)): 2, ((10, 6), (9, 71)): 3, ((10, 6), (10, 70)): 2, ((10, 5), (10, 49)): 2, ((10, 3), (10, 19)): 1, ((2, 3), (0, 8)): 1, ((2, 3), (0, 1)): 3, ((1, 3), (0, 8)): 3, ((4, 1), (0, 4)): 6, ((4, 1), (0, 2)): 4, ((4, 1), (10, 2)): 2, ((1, 1), (9, 5)): 1, ((1, 1), (4, 2)): 1, ((9, 5), (0, 9)): 3, ((9, 5), (9, 11)): 1, ((9, 5), (1, 2)): 1, ((2, 8), (0, 9)): 1, ((2, 8), (0, 1)): 1, ((5, 2), (0, 7)): 7, ((5, 2), (31, 3)): 1, ((5, 2), (10, 1)): 3, ((10, 5), (12, 8)): 2, ((10, 5), (5, 1)): 3, ((10, 0), (4, 7)): 1, ((10, 0), (2, 11)): 1, ((10, 5), (9, 51)): 2, ((9, 4), (10, 35)): 1, ((9, 4), (10, 7)): 2, ((9, 5), (10, 6)): 1, ((9, 5), (9, 6)): 1, ((1, 2), (10, 6)): 6, ((10, 4), (1, 4)): 1, ((10, 2), (5, 12)): 2, ((9, 5), (9, 30)): 1, ((10, 5), (1, 7)): 1, ((9, 2), (10, 14)): 2, ((9, 2), (1, 6)): 1, ((10, 6), (4, 6)): 1, ((10, 6), (9, 12)): 1, ((10, 3), (10, 55)): 4, ((10, 6), (7, 7)): 1, ((9, 0), (10, 4)): 4, ((9, 0), (9, 4)): 2, ((9, 0), (10, 7)): 3, ((10, 7), (8, 82)): 1, ((10, 7), (8, 86)): 1, ((1, 3), (0, 1)): 4, ((5, 0), (10, 27)): 1, ((5, 0), (0, 6)): 3, ((5, 0), (0, 2)): 1, ((10, 2), (8, 7)): 1, ((9, 3), (9, 6)): 2, ((10, 1), (18, 7)): 2, ((10, 4), (3, 11)): 1, ((10, 6), (10, 12)): 1, ((10, 6), (4, 4)): 1, ((9, 3), (4, 4)): 1, ((10, 0), (0, 13)): 3, ((10, 6), (12, 1)): 1, ((10, 6), (1, 4)): 4, ((10, 6), (28, 2)): 1, ((8, 7), (0, 15)): 1, ((8, 7), (10, 87)): 1, ((8, 7), (10, 73)): 1, ((8, 7), (10, 3)): 2, ((8, 7), (10, 5)): 1, ((8, 7), (0, 1)): 2, ((5, 6), (9, 3)): 1, ((5, 6), (10, 4)): 1, ((5, 6), (0, 1)): 2, ((5, 6), (1, 1)): 1, ((9, 2), (10, 81)): 6, ((9, 2), (0, 32)): 1, ((9, 2), (9, 56)): 2, ((12, 1), (6, 2)): 1, ((12, 1), (0, 11)): 1, ((12, 1), (0, 9)): 1, ((12, 1), (0, 1)): 2, ((12, 1), (2, 1)): 1, ((12, 1), (0, 4)): 1, ((12, 1), (10, 7)): 1, ((12, 1), (1, 2)): 2, ((12, 1), (0, 2)): 2, ((28, 2), (0, 4)): 2, ((28, 2), (9, 4)): 1, ((28, 2), (10, 72)): 1, ((28, 2), (9, 23)): 1, ((28, 2), (10, 6)): 1, ((28, 2), (0, 12)): 2, ((28, 2), (10, 10)): 1, ((28, 2), (9, 3)): 1, ((28, 2), (0, 7)): 1, ((28, 2), (0, 9)): 2, ((28, 2), (0, 10)): 1, ((28, 2), (0, 3)): 1, ((28, 2), (10, 4)): 1, ((28, 2), (10, 3)): 2, ((28, 2), (0, 2)): 5, ((28, 2), (0, 1)): 4, ((28, 2), (10, 2)): 1, ((10, 7), (13, 9)): 1, ((9, 5), (1, 5)): 2, ((9, 5), (0, 13)): 1, ((9, 5), (9, 1)): 1, ((10, 5), (9, 9)): 1, ((10, 2), (0, 32)): 2, ((5, 1), (0, 4)): 20, ((5, 1), (0, 1)): 17, ((5, 1), (1, 2)): 4, ((1, 2), (9, 71)): 1, ((10, 3), (3, 9)): 1, ((10, 3), (9, 18)): 2, ((10, 1), (10, 27)): 4, ((10, 1), (10, 41)): 9, ((10, 6), (9, 83)): 3, ((10, 6), (0, 32)): 1, ((10, 6), (0, 44)): 1, ((10, 1), (9, 56)): 8, ((10, 10), (4, 12)): 1, ((10, 10), (34, 9)): 1, ((10, 10), (9, 2)): 1, ((10, 1), (25, 5)): 2, ((9, 4), (9, 60)): 1, ((9, 4), (10, 81)): 5, ((9, 4), (0, 32)): 2, ((9, 4), (9, 56)): 2, ((9, 4), (10, 62)): 1, ((25, 5), (10, 74)): 1, ((25, 5), (9, 8)): 1, ((25, 5), (0, 5)): 2, ((25, 5), (10, 9)): 1, ((25, 5), (10, 6)): 1, ((25, 5), (10, 8)): 2, ((25, 5), (9, 4)): 1, ((25, 5), (10, 3)): 1, ((25, 5), (0, 4)): 2, ((25, 5), (0, 10)): 1, ((25, 5), (10, 11)): 1, ((25, 5), (10, 4)): 2, ((25, 5), (10, 7)): 1, ((25, 5), (0, 2)): 2, ((25, 5), (10, 2)): 1, ((25, 5), (9, 2)): 1, ((25, 5), (0, 3)): 2, ((25, 5), (0, 1)): 1, ((25, 5), (9, 3)): 1, ((9, 4), (9, 70)): 3, ((9, 4), (10, 16)): 1, ((9, 4), (3, 9)): 2, ((9, 2), (9, 7)): 3, ((10, 0), (10, 62)): 3, ((10, 0), (0, 25)): 1, ((10, 0), (10, 31)): 1, ((1, 0), (0, 4)): 1, ((7, 6), (0, 9)): 2, ((7, 6), (0, 3)): 1, ((7, 6), (0, 1)): 1, ((10, 2), (4, 9)): 1, ((10, 7), (6, 4)): 2, ((10, 7), (9, 1)): 1, ((10, 0), (12, 8)): 1, ((12, 8), (0, 7)): 1, ((12, 8), (0, 11)): 1, ((12, 8), (0, 3)): 1, ((12, 8), (0, 6)): 1, ((12, 8), (0, 5)): 3, ((12, 8), (0, 4)): 3, ((12, 8), (0, 1)): 1, ((12, 8), (6, 3)): 1, ((6, 3), (0, 13)): 2, ((6, 3), (1, 6)): 1, ((6, 3), (0, 5)): 1, ((6, 3), (0, 2)): 2, ((10, 2), (5, 6)): 2, ((10, 6), (4, 8)): 1, ((9, 1), (2, 3)): 1, ((9, 1), (1, 5)): 4, ((9, 1), (1, 6)): 1, ((9, 2), (0, 20)): 1, ((9, 2), (10, 12)): 1, ((10, 2), (1, 9)): 2, ((1, 2), (10, 2)): 1, ((10, 7), (10, 61)): 1, ((10, 7), (9, 70)): 1, ((10, 7), (0, 22)): 1, ((10, 7), (9, 2)): 2, ((9, 2), (0, 13)): 4, ((10, 3), (34, 9)): 1, ((10, 1), (9, 18)): 2, ((10, 3), (4, 8)): 1, ((10, 1), (9, 23)): 5, ((9, 3), (10, 35)): 2, ((9, 3), (0, 44)): 1, ((9, 2), (10, 13)): 5, ((9, 1), (10, 12)): 4, ((9, 1), (0, 11)): 4, ((9, 1), (10, 7)): 13, ((9, 1), (6, 4)): 1, ((9, 1), (10, 1)): 13, ((2, 3), (1, 3)): 1, ((9, 1), (10, 77)): 3, ((9, 1), (9, 60)): 9, ((9, 1), (9, 67)): 3, ((9, 1), (10, 56)): 8, ((9, 1), (10, 38)): 6, ((9, 1), (0, 67)): 6, ((9, 1), (10, 62)): 6, ((10, 1), (2, 1)): 3, ((2, 1), (0, 3)): 2, ((2, 1), (0, 6)): 2, ((10, 5), (4, 20)): 1, ((9, 0), (1, 4)): 2, ((9, 0), (1, 1)): 2, ((1, 1), (10, 6)): 5, ((9, 4), (0, 14)): 2, ((9, 4), (0, 18)): 1, ((9, 4), (9, 83)): 1, ((10, 4), (0, 22)): 2, ((9, 0), (0, 6)): 5, ((9, 0), (0, 7)): 2, ((9, 0), (10, 1)): 3, ((9, 0), (9, 1)): 2, ((9, 0), (0, 8)): 3, ((9, 1), (10, 59)): 8, ((9, 1), (8, 82)): 4, ((9, 1), (0, 91)): 5, ((9, 1), (8, 86)): 2, ((9, 1), (9, 63)): 2, ((9, 1), (10, 55)): 7, ((10, 0), (0, 8)): 6, ((10, 3), (9, 12)): 2, ((2, 1), (4, 2)): 1, ((2, 1), (1, 1)): 1, ((1, 2), (9, 2)): 1, ((1, 1), (0, 6)): 4, ((1, 1), (0, 8)): 3, ((10, 4), (10, 27)): 3, ((4, 3), (0, 7)): 1, ((4, 3), (0, 4)): 1, ((4, 3), (9, 5)): 1, ((9, 2), (9, 30)): 2, ((9, 2), (9, 70)): 3, ((9, 2), (10, 79)): 3, ((9, 2), (8, 2)): 1, ((5, 1), (0, 10)): 1, ((5, 1), (0, 2)): 17, ((34, 9), (10, 2)): 3, ((34, 9), (8, 13)): 1, ((34, 9), (0, 10)): 1, ((34, 9), (0, 1)): 26, ((34, 9), (10, 3)): 1, ((34, 9), (9, 4)): 1, ((34, 9), (0, 2)): 1, ((9, 3), (10, 57)): 1, ((9, 3), (9, 83)): 1, ((9, 2), (5, 2)): 1, ((10, 1), (9, 9)): 5, ((10, 5), (4, 12)): 1, ((2, 6), (3, 11)): 1, ((2, 6), (0, 1)): 1, ((9, 1), (10, 69)): 3, ((9, 1), (0, 50)): 2, ((9, 1), (9, 70)): 5, ((9, 1), (9, 83)): 5, ((10, 1), (10, 38)): 4, ((10, 0), (0, 54)): 1, ((10, 0), (9, 81)): 1, ((10, 0), (10, 57)): 1, ((9, 7), (10, 77)): 1, ((9, 7), (10, 60)): 1, ((9, 7), (10, 66)): 1, ((5, 2), (10, 7)): 2, ((10, 1), (0, 18)): 2, ((9, 1), (9, 1)): 4, ((9, 1), (9, 3)): 7, ((9, 6), (12, 8)): 1, ((9, 6), (10, 9)): 1, ((9, 1), (8, 6)): 3, ((10, 9), (9, 4)): 1, ((1, 3), (0, 3)): 2, ((10, 2), (4, 20)): 3, ((1, 9), (9, 2)): 1, ((9, 4), (7, 4)): 1, ((29, 4), (10, 3)): 5, ((29, 4), (9, 4)): 1, ((29, 4), (0, 9)): 1, ((29, 4), (0, 5)): 1, ((29, 4), (10, 4)): 2, ((29, 4), (10, 15)): 1, ((29, 4), (0, 4)): 4, ((29, 4), (9, 3)): 1, ((29, 4), (0, 6)): 2, ((29, 4), (0, 10)): 2, ((29, 4), (10, 5)): 1, ((29, 4), (0, 13)): 2, ((29, 4), (10, 13)): 1, ((29, 4), (1, 3)): 1, ((29, 4), (0, 2)): 1, ((29, 4), (9, 1)): 1, ((29, 4), (0, 1)): 2, ((9, 2), (5, 3)): 2, ((5, 1), (10, 3)): 9, ((10, 3), (2, 1)): 3, ((5, 1), (10, 9)): 3, ((5, 1), (10, 12)): 1, ((5, 1), (8, 7)): 1, ((10, 3), (9, 9)): 4, ((10, 3), (2, 3)): 3, ((1, 4), (0, 8)): 1, ((2, 3), (0, 12)): 1, ((2, 3), (10, 12)): 1, ((10, 5), (3, 9)): 1, ((9, 0), (0, 5)): 5, ((9, 0), (10, 6)): 1, ((10, 0), (7, 6)): 1, ((10, 0), (0, 14)): 2, ((10, 0), (8, 7)): 1, ((10, 0), (9, 4)): 2, ((9, 7), (0, 10)): 2, ((9, 7), (9, 6)): 1, ((9, 7), (0, 13)): 2, ((10, 14), (10, 85)): 1, ((10, 14), (10, 4)): 1, ((10, 14), (0, 4)): 1, ((10, 14), (9, 8)): 1, ((10, 14), (10, 1)): 1, ((10, 14), (0, 1)): 1, ((10, 2), (4, 2)): 1, ((18, 3), (9, 63)): 1, ((18, 3), (0, 5)): 1, ((18, 3), (10, 6)): 1, ((18, 3), (0, 4)): 2, ((18, 3), (0, 2)): 1, ((18, 3), (0, 3)): 1, ((18, 3), (9, 9)): 1, ((18, 3), (0, 1)): 10, ((9, 7), (9, 5)): 1, ((9, 0), (10, 13)): 1, ((9, 0), (9, 3)): 2, ((9, 3), (4, 9)): 1, ((1, 2), (0, 4)): 4, ((10, 0), (8, 13)): 1, ((10, 0), (18, 7)): 1, ((10, 0), (5, 7)): 1, ((10, 5), (6, 4)): 1, ((10, 3), (4, 1)): 2, ((4, 1), (1, 5)): 1, ((4, 1), (10, 4)): 1, ((4, 1), (0, 7)): 1, ((4, 1), (0, 5)): 1, ((5, 1), (34, 9)): 2, ((5, 1), (0, 8)): 5, ((9, 0), (10, 10)): 1, ((1, 2), (10, 13)): 1, ((1, 2), (9, 5)): 1, ((1, 2), (10, 11)): 1, ((6, 4), (0, 6)): 1, ((6, 4), (0, 7)): 1, ((6, 4), (0, 2)): 1, ((6, 4), (0, 8)): 1, ((10, 3), (10, 15)): 4, ((5, 4), (9, 3)): 1, ((5, 4), (10, 9)): 1, ((5, 4), (10, 3)): 2, ((5, 4), (10, 4)): 2, ((5, 4), (0, 1)): 3, ((10, 0), (9, 5)): 1, ((9, 5), (2, 3)): 1, ((10, 8), (10, 71)): 1, ((10, 8), (9, 47)): 1, ((10, 8), (9, 63)): 1, ((10, 8), (0, 52)): 1, ((10, 8), (7, 4)): 1, ((10, 0), (10, 67)): 1, ((10, 11), (10, 9)): 1, ((10, 11), (1, 6)): 1, ((10, 11), (9, 5)): 1, ((10, 11), (4, 12)): 1, ((10, 6), (10, 85)): 1, ((10, 1), (4, 12)): 2, ((10, 0), (5, 1)): 1, ((10, 4), (10, 49)): 4, ((19, 4), (10, 7)): 1, ((19, 4), (9, 83)): 1, ((19, 4), (10, 4)): 2, ((19, 4), (14, 11)): 1, ((19, 4), (0, 11)): 1, ((19, 4), (10, 6)): 1, ((19, 4), (0, 4)): 2, ((19, 4), (0, 6)): 1, ((19, 4), (5, 5)): 1, ((19, 4), (10, 2)): 2, ((19, 4), (9, 8)): 1, ((19, 4), (0, 3)): 1, ((19, 4), (1, 3)): 1, ((19, 4), (0, 2)): 1, ((19, 4), (1, 1)): 1, ((19, 4), (10, 1)): 1, ((9, 2), (8, 1)): 1, ((1, 4), (10, 3)): 1, ((5, 3), (0, 13)): 1, ((5, 3), (1, 4)): 1, ((10, 1), (10, 74)): 10, ((10, 1), (10, 85)): 11, ((10, 1), (0, 21)): 4, ((10, 1), (0, 19)): 5, ((9, 3), (9, 2)): 4, ((9, 1), (5, 1)): 4, ((10, 4), (5, 1)): 3, ((5, 1), (9, 5)): 2, ((5, 1), (10, 5)): 10, ((5, 1), (9, 4)): 1, ((5, 1), (5, 3)): 2, ((5, 1), (0, 13)): 2, ((9, 5), (0, 18)): 1, ((10, 6), (1, 5)): 1, ((9, 9), (0, 14)): 1, ((9, 9), (9, 5)): 1, ((9, 9), (9, 6)): 1, ((9, 9), (10, 6)): 1, ((9, 9), (0, 1)): 2, ((1, 3), (0, 5)): 1, ((10, 1), (29, 4)): 1, ((10, 1), (7, 4)): 1, ((7, 4), (0, 7)): 2, ((7, 4), (10, 9)): 1, ((7, 4), (0, 11)): 1, ((7, 4), (0, 15)): 1, ((7, 4), (9, 9)): 1, ((10, 3), (10, 38)): 5, ((9, 1), (0, 54)): 1, ((9, 1), (10, 33)): 2, ((9, 1), (9, 71)): 7, ((9, 1), (0, 52)): 6, ((10, 4), (4, 4)): 1, ((10, 3), (6, 3)): 1, ((1, 2), (0, 11)): 1, ((9, 3), (10, 87)): 1, ((9, 3), (0, 21)): 1, ((10, 1), (10, 59)): 6, ((10, 5), (3, 5)): 1, ((10, 4), (10, 19)): 1, ((9, 3), (10, 85)): 2, ((7, 0), (9, 23)): 1, ((7, 0), (10, 11)): 1, ((7, 0), (10, 15)): 1, ((7, 0), (10, 1)): 1, ((7, 0), (10, 4)): 1, ((7, 0), (9, 4)): 1, ((7, 0), (10, 9)): 1, ((10, 1), (8, 6)): 4, ((10, 4), (8, 5)): 2, ((9, 4), (10, 11)): 1, ((10, 5), (9, 10)): 1, ((10, 5), (18, 3)): 2, ((8, 5), (10, 5)): 1, ((8, 5), (10, 8)): 1, ((8, 5), (10, 3)): 1, ((8, 5), (5, 4)): 1, ((1, 1), (4, 7)): 1, ((9, 1), (10, 85)): 5, ((9, 2), (10, 20)): 1, ((9, 3), (9, 9)): 2, ((9, 2), (10, 78)): 2, ((9, 2), (10, 70)): 1, ((9, 2), (0, 52)): 1, ((5, 2), (0, 9)): 4, ((5, 2), (9, 2)): 1, ((9, 3), (10, 9)): 3, ((9, 3), (9, 12)): 1, ((10, 1), (12, 3)): 2, ((10, 2), (7, 3)): 1, ((10, 2), (7, 4)): 1, ((10, 2), (19, 4)): 1, ((10, 6), (9, 47)): 1, ((1, 1), (2, 3)): 1, ((10, 0), (0, 11)): 2, ((10, 9), (10, 59)): 1, ((10, 9), (10, 73)): 1, ((10, 9), (10, 72)): 2, ((10, 9), (9, 70)): 1, ((10, 9), (10, 67)): 1, ((10, 9), (10, 78)): 1, ((10, 9), (10, 70)): 1, ((10, 9), (0, 32)): 1, ((10, 9), (9, 60)): 1, ((10, 3), (4, 4)): 1, ((5, 1), (0, 5)): 13, ((5, 1), (0, 6)): 6, ((10, 1), (10, 60)): 7, ((1, 0), (8, 13)): 1, ((5, 1), (10, 8)): 1, ((1, 2), (0, 14)): 1, ((10, 0), (10, 78)): 1, ((10, 0), (10, 66)): 1, ((10, 0), (9, 56)): 1, ((10, 0), (0, 12)): 4, ((1, 1), (10, 1)): 4, ((10, 4), (3, 8)): 2, ((10, 1), (10, 13)): 5, ((5, 5), (10, 6)): 2, ((5, 5), (0, 8)): 1, ((5, 5), (10, 10)): 1, ((9, 8), (10, 85)): 1, ((9, 8), (10, 55)): 1, ((9, 8), (9, 8)): 1, ((9, 8), (10, 6)): 1, ((9, 8), (4, 4)): 1, ((1, 3), (10, 6)): 2, ((4, 4), (10, 3)): 1, ((4, 4), (8, 1)): 1, ((4, 4), (0, 3)): 4, ((4, 4), (0, 1)): 2, ((8, 1), (10, 14)): 1, ((8, 1), (0, 7)): 4, ((8, 1), (0, 6)): 3, ((8, 1), (5, 2)): 1, ((8, 1), (0, 1)): 9, ((7, 3), (10, 3)): 1, ((7, 3), (10, 1)): 1, ((7, 3), (0, 4)): 2, ((7, 3), (0, 2)): 1, ((7, 3), (0, 3)): 1, ((7, 3), (10, 2)): 1, ((10, 2), (0, 21)): 2, ((10, 6), (8, 1)): 1, ((1, 1), (8, 5)): 1, ((9, 4), (9, 47)): 1, ((9, 4), (0, 91)): 1, ((1, 2), (7, 6)): 1, ((2, 1), (0, 5)): 2, ((2, 1), (0, 1)): 3, ((9, 0), (10, 8)): 1, ((9, 0), (0, 22)): 1, ((9, 1), (10, 10)): 5, ((9, 1), (4, 3)): 1, ((4, 3), (0, 25)): 1, ((10, 1), (0, 67)): 6, ((10, 1), (9, 83)): 13, ((9, 7), (0, 12)): 1, ((9, 7), (0, 4)): 2, ((10, 1), (5, 12)): 4, ((4, 1), (10, 5)): 1, ((4, 1), (1, 2)): 1, ((10, 4), (2, 13)): 1, ((10, 0), (5, 2)): 1, ((5, 2), (0, 12)): 1, ((9, 1), (10, 57)): 2, ((9, 1), (1, 1)): 5, ((1, 1), (10, 5)): 1, ((8, 1), (9, 2)): 1, ((8, 1), (10, 3)): 5, ((8, 1), (10, 8)): 1, ((8, 1), (5, 3)): 1, ((8, 1), (9, 5)): 1, ((8, 1), (4, 2)): 1, ((9, 6), (10, 73)): 1, ((9, 6), (9, 67)): 1, ((9, 6), (10, 78)): 1, ((9, 6), (9, 47)): 1, ((9, 6), (9, 56)): 1, ((9, 6), (9, 60)): 1, ((9, 6), (10, 62)): 1, ((9, 6), (0, 44)): 1, ((9, 1), (1, 3)): 2, ((1, 0), (8, 1)): 1, ((8, 1), (1, 3)): 3, ((8, 1), (0, 8)): 2, ((10, 1), (4, 8)): 1, ((1, 3), (1, 2)): 1, ((10, 3), (2, 5)): 2, ((10, 3), (6, 7)): 3, ((5, 1), (1, 5)): 1, ((10, 0), (6, 1)): 1, ((6, 1), (0, 91)): 1, ((6, 1), (10, 5)): 1, ((6, 1), (10, 3)): 1, ((9, 3), (4, 20)): 1, ((10, 2), (4, 8)): 1, ((10, 1), (0, 25)): 3, ((9, 3), (10, 12)): 1, ((9, 3), (1, 9)): 1, ((5, 0), (0, 3)): 1, ((5, 0), (9, 4)): 1, ((5, 0), (10, 9)): 1, ((9, 2), (10, 67)): 4, ((9, 2), (10, 66)): 2, ((9, 2), (9, 14)): 1, ((9, 4), (10, 61)): 1, ((9, 4), (10, 71)): 1, ((9, 4), (0, 21)): 1, ((10, 5), (10, 71)): 2, ((10, 0), (47, 3)): 1, ((47, 3), (0, 3)): 5, ((47, 3), (0, 9)): 1, ((47, 3), (0, 5)): 4, ((47, 3), (0, 2)): 4, ((47, 3), (9, 23)): 1, ((47, 3), (10, 5)): 2, ((47, 3), (2, 11)): 1, ((47, 3), (10, 7)): 1, ((47, 3), (0, 10)): 1, ((47, 3), (10, 4)): 2, ((47, 3), (0, 11)): 1, ((47, 3), (9, 6)): 1, ((47, 3), (0, 6)): 2, ((47, 3), (10, 6)): 1, ((47, 3), (9, 18)): 1, ((47, 3), (10, 8)): 1, ((47, 3), (0, 4)): 2, ((47, 3), (0, 7)): 1, ((47, 3), (0, 1)): 10, ((47, 3), (10, 2)): 1, ((47, 3), (5, 2)): 1, ((47, 3), (1, 1)): 1, ((47, 3), (3, 2)): 1, ((47, 3), (10, 3)): 1, ((1, 1), (9, 1)): 1, ((3, 2), (0, 11)): 1, ((3, 2), (10, 6)): 1, ((3, 2), (0, 1)): 4, ((1, 0), (0, 1)): 1, ((10, 1), (14, 3)): 1, ((14, 3), (0, 10)): 1, ((14, 3), (10, 4)): 1, ((14, 3), (10, 3)): 3, ((14, 3), (8, 13)): 1, ((14, 3), (9, 2)): 1, ((14, 3), (0, 4)): 3, ((14, 3), (0, 3)): 2, ((14, 3), (1, 1)): 1, ((14, 3), (0, 1)): 1, ((10, 0), (10, 10)): 2, ((10, 0), (8, 3)): 1, ((8, 3), (0, 21)): 1, ((8, 3), (0, 12)): 1, ((8, 3), (0, 1)): 6, ((9, 5), (10, 4)): 2, ((9, 5), (2, 1)): 1, ((9, 2), (10, 15)): 2, ((5, 4), (0, 6)): 2, ((5, 4), (10, 6)): 2, ((5, 4), (0, 3)): 2, ((5, 4), (0, 7)): 1, ((1, 1), (1, 5)): 1, ((1, 0), (10, 1)): 1, ((1, 0), (9, 1)): 1, ((9, 1), (9, 51)): 2, ((9, 1), (9, 47)): 2, ((9, 1), (10, 66)): 5, ((9, 1), (10, 35)): 2, ((1, 3), (0, 11)): 1, ((10, 1), (2, 8)): 1, ((10, 3), (7, 4)): 1, ((10, 1), (9, 14)): 2, ((9, 0), (0, 15)): 1, ((9, 0), (10, 12)): 1, ((9, 0), (5, 2)): 1, ((5, 2), (10, 87)): 1, ((5, 2), (9, 3)): 3, ((9, 1), (7, 7)): 1, ((9, 1), (9, 2)): 2, ((9, 1), (9, 8)): 2, ((1, 5), (10, 7)): 1, ((1, 1), (9, 6)): 3, ((10, 3), (9, 21)): 4, ((10, 3), (14, 3)): 1, ((10, 0), (9, 18)): 1, ((10, 0), (8, 1)): 1, ((8, 1), (0, 2)): 2, ((8, 1), (9, 1)): 1, ((10, 1), (9, 63)): 11, ((10, 1), (8, 32)): 3, ((9, 2), (3, 8)): 1, ((9, 2), (9, 8)): 2, ((9, 2), (1, 3)): 4, ((9, 2), (8, 6)): 1, ((1, 3), (0, 7)): 1, ((10, 4), (4, 12)): 1, ((10, 1), (8, 9)): 1, ((8, 3), (0, 9)): 1, ((8, 3), (13, 7)): 1, ((8, 3), (10, 5)): 2, ((10, 2), (9, 8)): 3, ((10, 1), (1, 7)): 1, ((10, 1), (9, 51)): 5, ((1, 1), (10, 11)): 1, ((1, 1), (8, 6)): 1, ((9, 2), (8, 5)): 2, ((5, 4), (10, 10)): 1, ((5, 4), (0, 8)): 1, ((5, 4), (0, 4)): 2, ((5, 4), (10, 1)): 1, ((9, 0), (9, 7)): 1, ((10, 1), (8, 2)): 3, ((9, 7), (0, 15)): 1, ((9, 7), (10, 9)): 1, ((9, 7), (1, 2)): 1, ((9, 7), (5, 2)): 1, ((1, 3), (0, 2)): 2, ((8, 2), (10, 7)): 1, ((8, 2), (10, 2)): 1, ((8, 2), (10, 3)): 3, ((8, 2), (4, 1)): 1, ((1, 2), (10, 7)): 2, ((5, 2), (10, 13)): 1, ((9, 6), (0, 7)): 4, ((9, 6), (10, 13)): 1, ((9, 6), (9, 7)): 1, ((1, 1), (10, 9)): 2, ((10, 0), (10, 14)): 2, ((10, 0), (9, 11)): 1, ((8, 3), (10, 1)): 1, ((1, 1), (0, 4)): 13, ((1, 0), (0, 10)): 1, ((10, 2), (12, 8)): 2, ((1, 1), (0, 11)): 2, ((5, 2), (5, 12)): 1, ((5, 2), (9, 6)): 1, ((5, 2), (8, 3)): 1, ((5, 1), (9, 2)): 2, ((1, 1), (9, 71)): 1, ((2, 2), (0, 12)): 1, ((2, 2), (0, 8)): 1, ((10, 1), (6, 2)): 1, ((10, 2), (10, 14)): 9, ((10, 7), (9, 9)): 1, ((1, 4), (0, 13)): 1, ((3, 2), (0, 10)): 1, ((3, 2), (0, 3)): 1, ((5, 3), (10, 5)): 1, ((5, 3), (10, 2)): 1, ((5, 3), (5, 1)): 1, ((9, 3), (8, 6)): 1, ((10, 5), (4, 6)): 1, ((10, 5), (4, 3)): 1, ((9, 1), (10, 81)): 6, ((9, 1), (8, 32)): 2, ((9, 2), (2, 3)): 1, ((10, 1), (17, 3)): 1, ((10, 7), (1, 9)): 1, ((10, 7), (1, 5)): 1, ((10, 7), (5, 2)): 1, ((5, 2), (1, 1)): 2, ((1, 6), (0, 13)): 1, ((10, 1), (4, 4)): 2, ((8, 3), (10, 8)): 1, ((8, 3), (4, 7)): 1, ((10, 9), (0, 11)): 1, ((10, 9), (10, 15)): 1, ((9, 9), (7, 6)): 1, ((9, 9), (2, 2)): 1, ((9, 9), (0, 5)): 1, ((9, 9), (0, 4)): 1, ((10, 4), (7, 7)): 1, ((5, 4), (0, 2)): 3, ((5, 4), (9, 1)): 1, ((5, 2), (10, 72)): 1, ((5, 2), (10, 61)): 1, ((5, 2), (9, 60)): 1, ((5, 2), (8, 82)): 1, ((5, 2), (10, 81)): 1, ((8, 4), (0, 3)): 3, ((8, 4), (0, 4)): 2, ((8, 4), (10, 5)): 1, ((8, 4), (9, 3)): 2, ((8, 4), (0, 5)): 2, ((10, 2), (17, 3)): 1, ((1, 2), (10, 9)): 1, ((10, 3), (8, 5)): 1, ((9, 3), (14, 11)): 1, ((9, 3), (5, 6)): 1, ((9, 3), (9, 1)): 1, ((5, 2), (10, 4)): 2, ((5, 2), (10, 14)): 1, ((5, 2), (1, 2)): 2, ((4, 2), (10, 10)): 1, ((4, 2), (9, 6)): 1, ((4, 2), (0, 3)): 1, ((4, 4), (0, 5)): 1, ((4, 4), (0, 4)): 1, ((9, 3), (8, 82)): 3, ((8, 2), (10, 69)): 1, ((8, 2), (10, 87)): 1, ((8, 2), (10, 77)): 1, ((8, 2), (10, 73)): 1, ((8, 2), (10, 78)): 1, ((8, 2), (9, 70)): 2, ((8, 2), (10, 60)): 1, ((8, 2), (0, 10)): 3, ((9, 2), (9, 6)): 2, ((9, 6), (0, 15)): 1, ((9, 1), (10, 74)): 4, ((9, 1), (10, 49)): 3, ((9, 1), (10, 72)): 6, ((9, 1), (10, 70)): 1, ((9, 4), (0, 20)): 1, ((9, 4), (1, 4)): 1, ((10, 1), (4, 7)): 1, ((10, 1), (9, 10)): 4, ((10, 3), (8, 7)): 1, ((5, 1), (1, 4)): 1, ((10, 5), (0, 22)): 1, ((9, 4), (10, 14)): 1, ((9, 4), (9, 7)): 1, ((6, 4), (0, 5)): 1, ((6, 4), (10, 3)): 1, ((6, 4), (10, 4)): 1, ((6, 4), (0, 4)): 1, ((9, 3), (9, 4)): 1, ((10, 2), (8, 6)): 2, ((9, 3), (3, 8)): 1, ((9, 3), (10, 15)): 1, ((9, 3), (4, 8)): 1, ((1, 2), (1, 3)): 1, ((5, 2), (10, 20)): 1, ((5, 2), (9, 7)): 1, ((5, 2), (5, 5)): 1, ((10, 3), (2, 4)): 1, ((2, 2), (0, 7)): 1, ((2, 2), (0, 3)): 1, ((5, 1), (0, 12)): 2, ((10, 5), (10, 38)): 2, ((10, 5), (10, 31)): 2, ((5, 3), (0, 7)): 2, ((5, 3), (0, 12)): 1, ((10, 2), (8, 2)): 2, ((10, 1), (2, 2)): 1, ((10, 1), (10, 19)): 6, ((7, 2), (0, 2)): 2, ((7, 2), (47, 3)): 1, ((7, 2), (0, 3)): 1, ((7, 2), (9, 2)): 1, ((7, 2), (10, 2)): 1, ((7, 2), (0, 1)): 1, ((9, 2), (25, 5)): 1, ((9, 2), (4, 4)): 1, ((10, 5), (34, 9)): 1, ((10, 2), (4, 3)): 2, ((5, 2), (10, 33)): 1, ((5, 2), (0, 14)): 1, ((9, 1), (10, 11)): 1, ((9, 1), (0, 15)): 4, ((8, 2), (10, 5)): 2, ((8, 2), (13, 7)): 1, ((10, 1), (5, 7)): 1, ((9, 3), (2, 11)): 1, ((9, 3), (4, 7)): 1, ((9, 3), (7, 7)): 1, ((9, 3), (5, 2)): 1, ((10, 1), (9, 12)): 2, ((1, 2), (5, 3)): 1, ((5, 1), (10, 19)): 1, ((5, 1), (5, 5)): 2, ((5, 1), (9, 1)): 2, ((8, 2), (9, 60)): 1, ((8, 2), (10, 41)): 1, ((8, 2), (10, 81)): 2, ((8, 2), (9, 71)): 1, ((8, 2), (9, 83)): 1, ((10, 2), (3, 5)): 1, ((10, 1), (7, 6)): 2, ((9, 1), (10, 8)): 5, ((9, 1), (5, 5)): 2, ((10, 1), (6, 3)): 1, ((9, 2), (9, 18)): 2, ((9, 2), (3, 2)): 2, ((17, 3), (2, 13)): 1, ((17, 3), (8, 13)): 1, ((17, 3), (0, 4)): 1, ((17, 3), (10, 2)): 2, ((17, 3), (10, 5)): 1, ((17, 3), (10, 3)): 1, ((17, 3), (0, 3)): 3, ((17, 3), (9, 3)): 1, ((17, 3), (10, 1)): 1, ((17, 3), (0, 2)): 2, ((17, 3), (0, 1)): 1, ((17, 3), (1, 1)): 1, ((17, 3), (9, 2)): 1, ((9, 5), (10, 59)): 1, ((9, 5), (10, 49)): 2, ((9, 5), (10, 67)): 1, ((9, 5), (10, 79)): 1, ((9, 1), (10, 14)): 3, ((9, 1), (46, 1)): 1, ((10, 5), (10, 33)): 1, ((31, 3), (0, 10)): 3, ((31, 3), (0, 20)): 1, ((31, 3), (0, 9)): 1, ((31, 3), (0, 7)): 1, ((31, 3), (0, 6)): 3, ((31, 3), (10, 6)): 1, ((31, 3), (0, 5)): 1, ((31, 3), (0, 2)): 6, ((31, 3), (0, 3)): 3, ((31, 3), (0, 4)): 2, ((31, 3), (10, 2)): 1, ((31, 3), (0, 1)): 6, ((31, 3), (10, 1)): 2, ((9, 3), (10, 71)): 1, ((9, 3), (9, 63)): 1, ((1, 3), (0, 13)): 1, ((10, 5), (2, 13)): 1, ((5, 4), (1, 2)): 2, ((5, 4), (9, 5)): 1, ((5, 4), (0, 14)): 1, ((5, 4), (9, 7)): 1, ((5, 3), (9, 18)): 1, ((5, 3), (1, 1)): 1, ((10, 4), (3, 5)): 1, ((9, 1), (5, 3)): 3, ((9, 1), (4, 2)): 1, ((9, 1), (0, 44)): 1, ((10, 3), (9, 23)): 1, ((4, 8), (0, 9)): 1, ((4, 8), (0, 3)): 1, ((4, 8), (10, 3)): 1, ((4, 8), (0, 1)): 1, ((6, 1), (9, 4)): 1, ((6, 1), (9, 6)): 1, ((6, 1), (10, 4)): 2, ((6, 1), (0, 1)): 2, ((10, 1), (13, 9)): 3, ((9, 2), (10, 35)): 4, ((9, 2), (1, 7)): 1, ((9, 2), (5, 1)): 1, ((5, 1), (10, 74)): 1, ((5, 1), (0, 9)): 1, ((10, 1), (8, 13)): 4, ((9, 1), (10, 71)): 3, ((9, 1), (10, 67)): 2, ((9, 2), (10, 57)): 1, ((5, 2), (2, 2)): 1, ((1, 4), (9, 2)): 1, ((10, 2), (2, 4)): 1, ((9, 1), (8, 2)): 1, ((2, 1), (10, 9)): 1, ((2, 1), (0, 2)): 1, ((9, 2), (8, 13)): 1, ((10, 4), (29, 4)): 1, ((9, 1), (8, 13)): 2, ((9, 1), (3, 8)): 2, ((10, 2), (2, 1)): 1, ((10, 2), (6, 7)): 3, ((10, 4), (9, 11)): 1, ((5, 4), (10, 7)): 3, ((10, 1), (14, 11)): 2, ((10, 1), (17, 1)): 1, ((8, 1), (5, 1)): 1, ((8, 1), (3, 2)): 1, ((9, 2), (5, 6)): 1, ((10, 3), (8, 4)): 1, ((10, 3), (5, 7)): 1, ((1, 2), (1, 1)): 1, ((9, 3), (10, 33)): 1, ((9, 3), (10, 38)): 1, ((10, 5), (9, 18)): 1, ((5, 4), (1, 5)): 1, ((5, 4), (10, 5)): 1, ((12, 3), (10, 9)): 1, ((12, 3), (0, 6)): 1, ((12, 3), (0, 15)): 1, ((12, 3), (10, 2)): 1, ((12, 3), (0, 4)): 1, ((12, 3), (0, 5)): 1, ((12, 3), (10, 7)): 1, ((12, 3), (0, 3)): 1, ((12, 3), (0, 8)): 1, ((12, 3), (0, 2)): 1, ((12, 3), (0, 1)): 2, ((4, 4), (10, 5)): 1, ((4, 4), (1, 2)): 1, ((4, 3), (0, 5)): 1, ((4, 3), (0, 2)): 1, ((5, 1), (1, 7)): 1, ((8, 2), (0, 5)): 2, ((8, 2), (0, 6)): 1, ((8, 2), (9, 3)): 1, ((8, 2), (6, 7)): 1, ((5, 1), (2, 11)): 1, ((5, 1), (4, 8)): 1, ((9, 1), (9, 11)): 1, ((9, 1), (8, 3)): 1, ((9, 2), (14, 11)): 1, ((4, 2), (9, 4)): 1, ((4, 2), (10, 3)): 1, ((4, 2), (0, 6)): 1, ((10, 3), (1, 7)): 1, ((10, 1), (2, 11)): 2, ((6, 7), (10, 3)): 1, ((6, 7), (0, 5)): 1, ((6, 7), (0, 4)): 2, ((6, 7), (0, 1)): 1, ((6, 7), (0, 6)): 1, ((9, 3), (10, 20)): 1, ((10, 2), (8, 9)): 1, ((10, 3), (9, 14)): 2, ((1, 1), (8, 82)): 1, ((2, 1), (25, 5)): 1, ((10, 2), (7, 2)): 1, ((5, 3), (14, 11)): 1, ((5, 3), (9, 6)): 1, ((10, 5), (7, 6)): 1, ((8, 3), (0, 10)): 1, ((8, 3), (9, 2)): 1, ((1, 3), (10, 3)): 1, ((8, 6), (10, 8)): 1, ((8, 6), (0, 9)): 1, ((8, 6), (0, 4)): 1, ((8, 6), (10, 6)): 1, ((8, 6), (10, 5)): 1, ((9, 2), (9, 1)): 1, ((4, 1), (10, 1)): 2, ((4, 1), (9, 1)): 1, ((9, 1), (9, 10)): 1, ((9, 2), (10, 87)): 2, ((9, 4), (17, 2)): 1, ((9, 4), (5, 1)): 1, ((10, 2), (1, 7)): 1, ((5, 1), (9, 83)): 2, ((5, 1), (10, 1)): 3, ((5, 2), (0, 18)): 1, ((1, 2), (10, 14)): 1, ((4, 1), (10, 87)): 1, ((4, 1), (10, 78)): 1, ((4, 1), (9, 71)): 1, ((4, 1), (10, 81)): 1, ((9, 1), (9, 7)): 2, ((10, 4), (4, 6)): 1, ((2, 2), (0, 2)): 1, ((1, 1), (0, 13)): 3, ((9, 1), (0, 18)): 1, ((1, 1), (4, 8)): 1, ((10, 1), (11, 3)): 1, ((9, 2), (8, 4)): 2, ((46, 1), (8, 13)): 1, ((46, 1), (0, 10)): 2, ((46, 1), (0, 15)): 1, ((46, 1), (2, 11)): 1, ((46, 1), (10, 6)): 2, ((46, 1), (2, 4)): 1, ((46, 1), (10, 12)): 2, ((46, 1), (0, 7)): 2, ((46, 1), (1, 3)): 1, ((46, 1), (9, 2)): 1, ((46, 1), (0, 6)): 2, ((46, 1), (0, 4)): 9, ((46, 1), (0, 3)): 6, ((46, 1), (10, 7)): 2, ((46, 1), (10, 10)): 1, ((46, 1), (0, 9)): 1, ((46, 1), (0, 14)): 1, ((46, 1), (9, 5)): 2, ((46, 1), (0, 5)): 3, ((46, 1), (10, 5)): 1, ((46, 1), (10, 2)): 1, ((46, 1), (10, 3)): 1, ((46, 1), (9, 3)): 1, ((46, 1), (0, 1)): 1, ((8, 1), (1, 1)): 1, ((9, 1), (7, 4)): 1, ((9, 1), (9, 12)): 1, ((1, 2), (9, 4)): 1, ((9, 2), (10, 60)): 2, ((10, 3), (12, 3)): 1, ((5, 1), (0, 14)): 1, ((5, 1), (9, 6)): 1, ((9, 1), (10, 61)): 5, ((10, 3), (17, 2)): 1, ((5, 1), (0, 54)): 1, ((5, 1), (10, 61)): 1, ((5, 1), (9, 71)): 2, ((17, 1), (10, 7)): 1, ((17, 1), (0, 4)): 1, ((17, 1), (9, 60)): 1, ((17, 1), (0, 13)): 1, ((17, 1), (0, 7)): 2, ((17, 1), (0, 6)): 1, ((17, 1), (0, 2)): 4, ((17, 1), (10, 3)): 2, ((17, 1), (9, 3)): 2, ((17, 1), (0, 3)): 1, ((17, 1), (9, 1)): 1, ((3, 2), (0, 2)): 1, ((9, 1), (10, 13)): 1, ((1, 1), (12, 8)): 1, ((9, 3), (9, 71)): 1, ((9, 3), (5, 3)): 2, ((1, 1), (0, 9)): 3, ((9, 1), (10, 41)): 1, ((1, 2), (0, 20)): 1, ((1, 1), (9, 3)): 1, ((10, 3), (19, 4)): 1, ((10, 3), (6, 1)): 1, ((10, 3), (3, 2)): 1, ((9, 1), (2, 11)): 2, ((9, 1), (48, 1)): 1, ((3, 1), (0, 6)): 1, ((3, 1), (0, 2)): 1, ((3, 1), (10, 1)): 1, ((5, 2), (3, 2)): 1, ((8, 4), (10, 6)): 2, ((8, 4), (9, 4)): 1, ((8, 4), (10, 9)): 1, ((8, 4), (0, 1)): 2, ((1, 2), (4, 6)): 1, ((1, 2), (31, 3)): 1, ((17, 2), (9, 11)): 1, ((17, 2), (10, 10)): 1, ((17, 2), (0, 10)): 2, ((17, 2), (0, 4)): 5, ((17, 2), (0, 5)): 5, ((17, 2), (0, 2)): 3, ((17, 2), (10, 3)): 1, ((17, 2), (0, 3)): 3, ((17, 2), (0, 1)): 6, ((17, 2), (10, 1)): 1, ((5, 1), (0, 11)): 3, ((5, 1), (3, 9)): 1, ((5, 1), (4, 12)): 1, ((2, 1), (0, 7)): 1, ((2, 1), (10, 6)): 1, ((10, 3), (3, 11)): 1, ((4, 1), (0, 3)): 1, ((4, 1), (10, 3)): 1, ((1, 1), (10, 12)): 1, ((1, 1), (1, 2)): 1, ((10, 6), (2, 11)): 1, ((1, 1), (0, 44)): 1, ((10, 1), (4, 1)): 2, ((10, 1), (34, 9)): 1, ((9, 1), (10, 9)): 2, ((9, 1), (5, 4)): 2, ((8, 2), (1, 2)): 1, ((8, 2), (0, 2)): 1, ((8, 2), (1, 1)): 1, ((17, 2), (0, 6)): 2, ((17, 2), (0, 9)): 2, ((17, 2), (0, 8)): 1, ((17, 2), (9, 6)): 1, ((9, 1), (10, 15)): 1, ((9, 1), (3, 5)): 1, ((6, 1), (1, 1)): 1, ((6, 1), (5, 1)): 1, ((3, 2), (10, 69)): 1, ((3, 2), (0, 21)): 1, ((3, 2), (10, 81)): 1, ((48, 1), (10, 74)): 1, ((48, 1), (10, 3)): 3, ((48, 1), (0, 11)): 2, ((48, 1), (10, 4)): 1, ((48, 1), (5, 12)): 1, ((48, 1), (9, 47)): 1, ((48, 1), (9, 7)): 1, ((48, 1), (0, 2)): 6, ((48, 1), (10, 8)): 1, ((48, 1), (0, 9)): 1, ((48, 1), (10, 2)): 2, ((48, 1), (0, 8)): 1, ((48, 1), (9, 6)): 1, ((48, 1), (9, 3)): 3, ((48, 1), (10, 5)): 3, ((48, 1), (34, 9)): 1, ((48, 1), (0, 3)): 2, ((48, 1), (10, 6)): 2, ((48, 1), (0, 5)): 1, ((48, 1), (0, 4)): 1, ((48, 1), (0, 7)): 1, ((48, 1), (10, 14)): 1, ((48, 1), (5, 4)): 1, ((48, 1), (1, 2)): 1, ((48, 1), (10, 1)): 3, ((48, 1), (0, 1)): 5, ((48, 1), (1, 1)): 1, ((5, 1), (10, 77)): 1, ((5, 1), (9, 70)): 1, ((5, 1), (10, 57)): 1, ((5, 1), (8, 82)): 1, ((5, 1), (10, 62)): 2, ((2, 1), (0, 4)): 1, ((1, 1), (10, 72)): 1, ((10, 1), (8, 5)): 1, ((10, 2), (8, 4)): 1, ((10, 1), (8, 1)): 2, ((4, 1), (1, 1)): 1, ((9, 2), (10, 16)): 1, ((9, 2), (1, 5)): 1, ((10, 2), (2, 2)): 1, ((10, 2), (4, 7)): 1, ((10, 1), (2, 6)): 1, ((10, 1), (9, 21)): 1, ((1, 1), (10, 69)): 1, ((9, 1), (5, 2)): 1, ((5, 1), (10, 81)): 1, ((5, 1), (10, 15)): 1, ((9, 3), (7, 6)): 1, ((10, 1), (6, 4)): 1, ((10, 2), (41, 1)): 1, ((9, 1), (9, 9)): 1, ((8, 1), (0, 18)): 1, ((9, 1), (8, 4)): 1, ((9, 1), (9, 23)): 1, ((41, 1), (0, 4)): 2, ((41, 1), (10, 9)): 1, ((41, 1), (10, 31)): 1, ((41, 1), (0, 6)): 1, ((41, 1), (0, 5)): 1, ((41, 1), (0, 12)): 1, ((41, 1), (0, 3)): 2, ((41, 1), (0, 1)): 32, ((10, 1), (10, 20)): 1, ((2, 1), (5, 2)): 1, ((8, 1), (10, 4)): 1, ((8, 1), (2, 1)): 1, ((10, 1), (31, 3)): 1, ((10, 1), (17, 2)): 1}
print("原图：")
EG.evaluate(g)
print("分组前结果：")
for i in range(5):
    node_list,new_JDM = Adapt(JDM)
    newIndegSeq = getIndegreeSeq(node_list)
    newOutdegSeq = getOutdegreeSeq(node_list)
    newNkk = GO.get_nkk(new_JDM)
    G = DK.directed_joint_degree_model(newIndegSeq, newOutdegSeq, newNkk)
    print("未加噪声生成图，第",i+1,"轮：")
    EG.evaluate(G)
    sortlist = [40, 110, 180]
    print("epsilon=200，第", i + 1, "轮：")
    regularper(JDM, 200)
    print("epsilon=100，第", i + 1, "轮：")
    regularper(JDM, 100)
    print("epsilon=50，第",i+1,"轮：")
    regularper(JDM, 50)
    print("epsilon=30，第",i+1,"轮：")
    regularper(JDM, 30)
    print("epsilon=20，第",i+1,"轮：")
    regularper(JDM, 20)
    print("epsilon=10，第",i+1,"轮：")
    regularper(JDM, 10)
    print("epsilon=5，第",i+1,"轮：")
    regularper(JDM, 5)
    print("epsilon=1，第", i + 1, "轮：")
    regularper(JDM, 1)
epsilon_list=[1,5,10,20,30,50,100,200]
sortlist=[1,8,30]
print("分组后结果：")
for i in epsilon_list:
    for j in range(5):
        print("epsilon=",i,"，第",j+1,"轮：")
        G = groupingPer(g, sortlist,i)
        print(G)
