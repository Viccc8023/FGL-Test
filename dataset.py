import os
import torch
import numpy as np
import networkx as nx
from torch.utils.data import random_split


# ==========================================
# 1. 统一的图数据容器
# ==========================================
class GraphData:
    def __init__(self, edge_mat, node_features, label):
        self.edge_mat = edge_mat
        self.node_features = node_features
        self.label = label
        self.num_nodes = edge_mat.shape[0]


# ==========================================
# 2. 数据加载器基类
# ==========================================
class BaseDatasetLoader:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name

    def load(self):
        raise NotImplementedError("子类必须实现 load() 方法")


# ==========================================
# 3. 专属的数据加载器 (支持标准 TU Dataset 及混合格式)
# ==========================================
class StandardTULoader(BaseDatasetLoader):
    def __init__(self, dataset_name):
        super().__init__(dataset_name)
        self.data_dir = f"./dataset/{self.dataset_name}"

    def load(self):
        print(f"正在加载数据集: {self.dataset_name}...")

        # ==========================================
        # 兼容不同的文件命名格式 (精准匹配 COLLAB)
        # ==========================================
        if self.dataset_name == 'COLLAB':
            edges_path = os.path.join(self.data_dir, f"{self.dataset_name}_edges.txt")
            indicator_path = os.path.join(self.data_dir, f"{self.dataset_name}_graph_idx.txt")
            graph_labels_path = os.path.join(self.data_dir, f"{self.dataset_name}_graph_labels.txt")
        else:
            edges_path = os.path.join(self.data_dir, f"{self.dataset_name}_A.txt")
            indicator_path = os.path.join(self.data_dir, f"{self.dataset_name}_graph_indicator.txt")
            graph_labels_path = os.path.join(self.data_dir, f"{self.dataset_name}_graph_labels.txt")

        # ==========================================
        # 第一步：读取图结构 (所有数据集通用)
        # ==========================================
        # 1. 读取图标签
        with open(graph_labels_path, 'r') as f:
            graph_labels = [int(line.strip()) for line in f.readlines()]

        num_graphs = len(graph_labels)
        nx_graphs = [nx.Graph() for _ in range(num_graphs)]

        # 2. 读取节点所属图的索引 (TU Dataset 全局节点从 1 开始编号)
        with open(indicator_path, 'r') as f:
            node_to_graph_id = [int(line.strip()) - 1 for line in f.readlines()]

        num_total_nodes = len(node_to_graph_id)
        global_to_local_node_id = {}
        local_node_counter = [0] * num_graphs

        for global_id_minus_1, g_idx in enumerate(node_to_graph_id):
            global_id = global_id_minus_1 + 1
            local_id = local_node_counter[g_idx]
            global_to_local_node_id[global_id] = local_id
            nx_graphs[g_idx].add_node(local_id)
            local_node_counter[g_idx] += 1

        # 3. 读取边并建图
        with open(edges_path, 'r') as f:
            for line in f.readlines():
                u, v = map(int, line.strip().replace(',', ' ').split())
                g_idx = node_to_graph_id[u - 1]
                local_u = global_to_local_node_id[u]
                local_v = global_to_local_node_id[v]
                nx_graphs[g_idx].add_edge(local_u, local_v)

        # ==========================================
        # 第二步：处理节点特征 (用 IF 区分不同数据集)
        # ==========================================
        print("正在处理节点特征...")
        global_node_features = None

        # [情况 A]: 没有节点特征文件，使用度数(Degree)作为特征 (包含 REDDIT 和 COLLAB)
        if self.dataset_name in ['REDDIT-MULTI-5K', 'COLLAB']:
            pass  # 具体计算逻辑放在第三步循环中

        # [情况 B]: 仅有离散的节点标签，需转为 One-hot
        elif self.dataset_name in ['MUTAG', 'DD']:
            labels_path = os.path.join(self.data_dir, f"{self.dataset_name}_node_labels.txt")
            with open(labels_path, 'r') as f:
                n_labels = [int(line.strip()) for line in f.readlines()]

            max_label = max(n_labels)
            global_node_features = torch.zeros(num_total_nodes, max_label + 1)
            for i, l in enumerate(n_labels):
                global_node_features[i, l] = 1.0

        # [情况 C]: 既有标签又有连续属性，拼在一起
        elif self.dataset_name in ['PROTEINS']:
            labels_path = os.path.join(self.data_dir, f"{self.dataset_name}_node_labels.txt")
            attrs_path = os.path.join(self.data_dir, f"{self.dataset_name}_node_attributes.txt")

            with open(labels_path, 'r') as f:
                n_labels = [int(line.strip()) for line in f.readlines()]
            max_label = max(n_labels)

            with open(attrs_path, 'r') as f:
                n_attrs = [[float(x) for x in line.strip().split(',')] for line in f.readlines()]

            attr_dim = len(n_attrs[0])
            global_node_features = torch.zeros(num_total_nodes, max_label + 1 + attr_dim)

            for i, (l, attr) in enumerate(zip(n_labels, n_attrs)):
                global_node_features[i, l] = 1.0
                global_node_features[i, max_label + 1:] = torch.tensor(attr)

        else:
            raise ValueError(f"未配置此数据集的特征处理规则: {self.dataset_name}")

        # ==========================================
        # 第三步：打包成神经网络需要的 GraphData
        # ==========================================
        graph_list = []

        # 仅针对情况 A (无特征的图) 计算全图最大度数
        max_degree = 0
        if self.dataset_name in ['REDDIT-MULTI-5K', 'COLLAB']:
            for g in nx_graphs:
                if g.number_of_nodes() > 0:
                    degrees = [d for n, d in g.degree()]
                    if degrees:  # 防止空图报错
                        max_degree = max(max_degree, max(degrees))

        for g_idx, g in enumerate(nx_graphs):
            num_nodes = g.number_of_nodes()
            if num_nodes == 0:
                continue

                # 1. 生成邻接矩阵 Tensor (内存极致优化版)
            edge_mat = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
            edges = list(g.edges())
            if len(edges) > 0:
                u, v = zip(*edges)
                edge_mat[list(u), list(v)] = 1.0
                edge_mat[list(v), list(u)] = 1.0

            # 2. 分配特征
            if self.dataset_name in ['REDDIT-MULTI-5K', 'COLLAB']:
                # 情况 A：现场计算 One-hot 度数
                node_features = torch.zeros(num_nodes, max_degree + 1)
                for idx, deg in g.degree():
                    node_features[idx, deg] = 1.0
            else:
                # 情况 B 和 C：从全局特征提取
                feat_dim = global_node_features.shape[1]
                node_features = torch.zeros(num_nodes, feat_dim)
                for global_id, local_id in global_to_local_node_id.items():
                    if node_to_graph_id[global_id - 1] == g_idx:
                        node_features[local_id] = global_node_features[global_id - 1]

            # 3. 压入容器
            graph_list.append(GraphData(edge_mat, node_features, graph_labels[g_idx]))

        print(f"成功加载 {len(graph_list)} 张图！")
        return graph_list


# ==========================================
# 4. 调度中心
# ==========================================
def get_dataset(dataset_name):
    # 现在这四个数据集统一由 StandardTULoader 处理
    loader = StandardTULoader(dataset_name)
    return loader.load()


# ==========================================
# 5. 联邦学习切分功能
# ==========================================
def split_federated_data(graph_list, num_clients):
    total_size = len(graph_list)
    test_size = int(total_size * 0.2)
    train_size = total_size - test_size
    train_data, test_data = random_split(graph_list, [train_size, test_size])

    client_data_sizes = [train_size // num_clients] * num_clients
    client_data_sizes[0] += train_size % num_clients
    clients_data = random_split(train_data, client_data_sizes)
    return clients_data, test_data



# === 测试模块 ===
if __name__ == "__main__":
    # 你可以把 MUTAG 改成 DD 或 PROTEINS 或 REDDIT-MULTI-5K 来测试不同的分支
    test_dataset = 'COLLAB'
    num_clients = 3  # 设定联邦学习的客户端数量

    print(f"--- 启动 {test_dataset} 模块测试 ---")

    # 确保当前目录下有 dataset/MUTAG/ 并且包含真实数据文件
    try:
        # 1. 加载数据
        graphs = get_dataset(test_dataset)

        # 2. 联邦切分数据
        clients_data, test_data = split_federated_data(graphs, num_clients=num_clients)

        # 3. 打印统计信息
        print(f"\n================ 测试结果 ================")
        print(f"数据集总图数: {len(graphs)}")
        print(f"用于全局测试的图数量 (20%): {len(test_data)}")

        print(f"\n--- 客户端数据分配详情 ---")
        for i, client_subset in enumerate(clients_data):
            print(f"客户端 {i + 1} 分配到的图数量: {len(client_subset)}")

        print(f"\n--- 数据结构抽查 ---")
        sample_graph = graphs[0]
        print(f"抽查第 1 张图 -> 节点数: {sample_graph.num_nodes}, 特征矩阵维度: {sample_graph.node_features.shape}")
        print(f"==========================================")

    except Exception as e:
        print(f"报错啦！请检查数据集 {test_dataset} 的文件是否放对了位置: {e}")