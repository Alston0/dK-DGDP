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



g = nx.read_edgelist("D:\\pythonversion\\data\\p2p-Gnutella08.txt", create_using=nx.DiGraph)
JDM=GO.getQuintuple(g)
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
