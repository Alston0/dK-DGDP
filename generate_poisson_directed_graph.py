import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random

def generate_poisson_directed_graph(num_nodes=100, num_edges=500, poisson_lambda=5, seed=None):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    # Step 1: 生成泊松分布的出入度
    out_degrees = np.random.poisson(poisson_lambda, num_nodes)
    in_degrees = np.random.poisson(poisson_lambda, num_nodes)

    # Step 2: 调整到目标边数
    scale = num_edges / max(sum(out_degrees), sum(in_degrees), 1e-5)
    out_degrees = np.round(out_degrees * scale).astype(int)
    in_degrees = np.round(in_degrees * scale).astype(int)

    while sum(out_degrees) > num_edges:
        i = np.argmax(out_degrees)
        if out_degrees[i] > 0:
            out_degrees[i] -= 1
    while sum(in_degrees) > num_edges:
        i = np.argmax(in_degrees)
        if in_degrees[i] > 0:
            in_degrees[i] -= 1

    # Step 3: 构造端点列表
    out_stubs = []
    in_stubs = []
    for i in range(num_nodes):
        out_stubs.extend([i] * out_degrees[i])
        in_stubs.extend([i] * in_degrees[i])

    random.shuffle(out_stubs)
    random.shuffle(in_stubs)

    G = nx.DiGraph()
    G.add_nodes_from(range(num_nodes))
    edge_set = set()

    for u, v in zip(out_stubs, in_stubs):
        if u != v and (u, v) not in edge_set:
            G.add_edge(u, v)
            edge_set.add((u, v))
        if G.number_of_edges() >= num_edges:
            break

    print(f"图生成完成：节点数 = {G.number_of_nodes()}, 边数 = {G.number_of_edges()}")
    return G

def save_graph_to_txt(G, filepath="poisson_directed_graph.txt"):
    with open(filepath, "w") as f:
        for u, v in G.edges():
            f.write(f"{u} {v}\n")
    print(f"图已保存为：{filepath}")

def visualize_graph(G):
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, node_size=50, arrowsize=10, with_labels=False)
    plt.title("Directed Graph with Poisson Degree Distribution")
    plt.show()

if __name__ == "__main__":
    # 参数设置
    num_nodes = 1800      # 节点数
    num_edges = 10000      # 边数
    poisson_lambda = 3   # 泊松分布参数λ
    seed = 42            # 随机种子

    # 生成图
    G = generate_poisson_directed_graph(num_nodes, num_edges, poisson_lambda, seed)

    # 保存为 txt
    save_graph_to_txt(G, "poisson_directed_graph.txt")

    # 可视化（可选）
    # visualize_graph(G)
