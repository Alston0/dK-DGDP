from collections import Counter
import networkx as nx
import math
import numpy as np
import random

def searchDK2(seq, degree):
    matr = np.zeros([21120, 21120]).astype(int)
    for i in range(len(seq)):
        for j in seq[i]:
            ind1 = degree[i].astype(int)
            ind2 = degree[j].astype(int)
            matr[ind1, ind2] += 1
    return matr

def showDK2(matr):
    for i in range(len(matr)):
        for j in range(len(matr[i])):
            if matr[i, j]:
                print(i, j, matr[i, j])

def Add_noise(degree_vector, delta_f, epsilon):
    """
    :param degree_vector: 度向量
    :param delta_f: 敏感度
    :param epsilon: 隐私预算
    """
    noised_degree_vector = degree_vector + np.random.laplace(loc=0, scale=(delta_f / epsilon), size=degree_vector.shape)
    noised_degree_vector = np.round(noised_degree_vector)
    for i in range(len(noised_degree_vector)):
        for j in range(len(noised_degree_vector[i])):
            if noised_degree_vector[i, j] < 0:
                noised_degree_vector[i, j] = 0
    return noised_degree_vector
def node_degree_joint(G):
    for i in range(len(list(G.edges()))):
        degu=G.degree[list(G.edges())[i][0]]
        degv=G.degree[list(G.edges())[i][1]]
        yield (degu,degv)
def gettrituple(G):
    JDAM={}
    edgesdegree_list = list(node_degree_joint(G))
    for i in range(len(edgesdegree_list)):
        if edgesdegree_list[i] in JDAM:
            JDAM[edgesdegree_list[i]]+=1
        elif (edgesdegree_list[i][1],edgesdegree_list[i][0]) in JDAM:
            JDAM[(edgesdegree_list[i][1],edgesdegree_list[i][0])] += 1
        else:
            JDAM[edgesdegree_list[i]]=JDAM.get((edgesdegree_list[i]),0)+1
    return JDAM

def perJDM(JDM,epsilon):
    maxDegree = 0
    DpJDM_value = []
    numOfZero = 0
    valsLessThanZero = 0
    Dp=JDM.copy()
    JDM_items = list(JDM.items())
    for i in range(len(JDM_items)):
        if (JDM_items[i][0][0] > maxDegree):
            maxDegree = JDM_items[i][0][0]
        if (JDM_items[i][0][1] > maxDegree):
            maxDegree = JDM_items[i][0][1]
    sensitivity = 4*maxDegree+1
    absPositiveSum = 0
    absNegativeSum = 0
    while (absNegativeSum >= absPositiveSum):
        JDM_values = list(JDM.values())
        Laplacenoisy = np.random.laplace(loc=0, scale=(sensitivity / epsilon), size=len(JDM_values))
        DpJDM_value = JDM_values + Laplacenoisy
        for i in range(len(DpJDM_value)):
            newVal = DpJDM_value[i]
            if newVal < 0:
                absNegativeSum += abs(newVal)
                valsLessThanZero += -1 * newVal
                numOfZero += 1
                DpJDM_value[i] = 0
                newVal = 0
            else:
                DpJDM_value[i] = np.round(newVal)
                absPositiveSum += newVal
            Dp[JDM_items[i][0]] = np.round(newVal)
    iterSum = 0
    cutValue = 0
    iterCount = 0
    for i in range(len(DpJDM_value)):
        iterSum += ((DpJDM_value[i] - cutValue) * (len(DpJDM_value) - iterCount))
        if (iterSum > absNegativeSum):
            break
        cutValue = DpJDM_value[i]
        iterCount += 1
    if iterCount != len(DpJDM_value):
        sub = (absNegativeSum - iterSum) / (len(DpJDM_value) - iterCount)
        for i in JDM.keys():
            if Dp[i] <= cutValue:
                Dp[i] = Dp[i] - cutValue - sub
    return Dp

def perJDAM(JDAM,epsilon):
    maxDegree = 0
    DpJDAM_value = []
    numOfZero = 0
    valsLessThanZero = 0
    Dp = JDAM.copy()
    JDAM_items = list(JDAM.items())
    for i in range(len(JDAM_items)):
        if (JDAM_items[i][0][0] > maxDegree):
            maxDegree = JDAM_items[i][0][0]
        if (JDAM_items[i][0][1] > maxDegree):
            maxDegree = JDAM_items[i][0][1]
    sensitivity = 4 * maxDegree + 1
    absPositiveSum = 0
    absNegativeSum = 0
    while (absNegativeSum >= absPositiveSum):
        JDAM_values = list(JDAM.values())
        Laplacenoisy = np.random.laplace(loc=0, scale=(sensitivity / epsilon), size=len(JDAM_values))
        DpJDAM_value = JDAM_values + Laplacenoisy
        for i in range(len(DpJDAM_value)):
            newVal = DpJDAM_value[i]
            if newVal < 0:
                absNegativeSum += abs(newVal)
                valsLessThanZero += -1 * newVal
                numOfZero += 1
                DpJDAM_value[i] = 0
                newVal = 0
            else:
                DpJDAM_value[i] = np.round(newVal).astype(int)
                absPositiveSum += newVal
            Dp[JDAM_items[i][0]] = np.round(newVal).astype(int)
    print(DpJDAM_value)
    print(Dp)
    return Dp


# p2p-Gnutella08.txt
# Wiki-Vote.txt
# example.txt
# soc-Wiki-Vote.txt
g = nx.read_edgelist("D:\\pythonversion\\data\\example.txt" , create_using=nx.Graph)
print(g)
JDAM=gettrituple(g)
print(JDAM)
DpJDAM=perJDAM(JDAM,50)
nodecount={}
edgecount={}
neibor={}
V_list={}
DpJDAM_items=list(DpJDAM.items())
for i in range(len(DpJDAM_items)):
    edgecount[DpJDAM_items[i][0][0]]=edgecount.get(DpJDAM_items[i][0][0],0)+DpJDAM_items[i][1]
    edgecount[DpJDAM_items[i][0][1]] = edgecount.get(DpJDAM_items[i][0][1], 0) + DpJDAM_items[i][1]
    neibor.setdefault(DpJDAM_items[i][0][0],{})
    neibor.setdefault(DpJDAM_items[i][0][1], {})
    neibor[DpJDAM_items[i][0][0]][DpJDAM_items[i][0][1]]=neibor[DpJDAM_items[i][0][0]].get(DpJDAM_items[i][0][1],0)+DpJDAM_items[i][1]
    neibor[DpJDAM_items[i][0][1]] [DpJDAM_items[i][0][0]]= neibor[DpJDAM_items[i][0][1]].get(DpJDAM_items[i][0][0], 0) + DpJDAM_items[i][1]
print(neibor)
print(edgecount)
for i in edgecount:
    nodecount[i]=nodecount.get(i,0)+round(edgecount[i]/i,2)
print(nodecount)
for i in nodecount:
    #处理度为i的节点
    if nodecount[i]%1==0:
        V_list[i]=V_list.get(i,0)+int(nodecount[i])
        newdegree=0
    elif nodecount[i]<1.5:
        V_list[edgecount[i]] = V_list.get(edgecount[i], 0) + 1
        newdegree=edgecount[i]
    else:
        integer = int(nodecount[i])
        V_list[i] = V_list.get(i, 0) + int(nodecount[i]) - 1
        decimal=nodecount[i]-int(nodecount[i])+1
        if decimal < 1.5:
            newdegree=edgecount[i]-(int(nodecount[i])-1)*i
        else:
            V_list[i] = V_list.get(i, 0) + 1
            newdegree=edgecount[i]-int(nodecount[i])*i
        V_list[newdegree] = V_list.get(newdegree, 0) + 1
    j=0
    while j<newdegree:
        ranchoice=random.sample(neibor[i].keys(),1)
        node=ranchoice[0]
        if neibor[node][i]>0:
            if (node,newdegree) in DpJDAM:
                DpJDAM[(node, newdegree)]+=1
            elif (newdegree,node) in DpJDAM:
                DpJDAM[(newdegree,node)]+=1
            else:
                DpJDAM[(node,newdegree)]=DpJDAM.get((node,newdegree),0)+1
            if (node,i) in DpJDAM:
                DpJDAM[(node,i)]-=1
            elif (i,node) in DpJDAM:
                DpJDAM[(i,node)] -= 1
            neibor[node][i]-=1
            neibor[node][newdegree]=neibor[node].get(newdegree,0)+1

            neibor[i][node]-=1
            neibor.setdefault(newdegree,{})
            neibor[newdegree][node]=neibor[newdegree].get(node,0)+1
            j+=1
        else:
            continue
print(V_list)
print(DpJDAM)
print(neibor)
G = nx.joint_degree_graph(neibor)
print(G)
