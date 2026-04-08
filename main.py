import argparse
import copy
import random
import os
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# 从你提供的项目文件中导入模块
from dataset import get_dataset, split_federated_data
from cnn import GraphCNN

# ==========================================
# 0. 全局设备检测
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================
# 1. 客户端定义 (Client)
# ==========================================
class Client:
    def __init__(self, client_id, local_data, is_malicious, input_dim, output_dim, dataset_name):
        self.client_id = client_id
        self.local_data = local_data
        self.is_malicious = is_malicious
        self.dataset_name = dataset_name

        # 初始化本地模型并传送到设备
        self.local_model = GraphCNN(
            num_layers=3,
            num_mlp_layers=2,
            input_dim=input_dim,
            hidden_dim=64,
            output_dim=output_dim,
            final_dropout=0.5,
            dataset_name=dataset_name
        ).to(device)

        self.criterion = nn.CrossEntropyLoss()

    def train(self, global_weights, local_epochs, attack_method, current_round):
        self.local_model.load_state_dict(copy.deepcopy(global_weights))
        self.local_model.train()

        # 【针对 COLLAB 的优化】：降低基础学习率以防止梯度爆炸
        base_lr = 0.001 if self.dataset_name in ['COLLAB', 'DD', 'PROTEINS'] else 0.01
        lr = base_lr * (0.5 ** (current_round // 50))

        # 仿照 Opt-GDBA：每轮重新初始化优化器以消除动量干扰
        optimizer = optim.Adam(self.local_model.parameters(), lr=lr)

        if self.is_malicious and attack_method != 'no_attack':
            # 攻击逻辑预留入口
            return copy.deepcopy(self.local_model.state_dict()), 0.0
        else:
            total_loss = 0.0
            for _ in range(local_epochs):
                # 随机采样一个 Batch
                batch_size = min(128, len(self.local_data))
                selected_idx = np.random.permutation(len(self.local_data))[:batch_size]
                batch_graph = [self.local_data[idx] for idx in selected_idx]
                batch_labels = torch.tensor([g.label for g in batch_graph], dtype=torch.long).to(device)

                optimizer.zero_grad()
                logits = self.local_model(batch_graph)
                loss = self.criterion(logits, batch_labels)
                loss.backward()

                # 【救命稻草】：梯度裁剪，防止权重飞向无穷大
                torch.nn.utils.clip_grad_norm_(self.local_model.parameters(), max_norm=2.0)

                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / local_epochs if local_epochs > 0 else 0.0
            return copy.deepcopy(self.local_model.state_dict()), avg_loss


# ==========================================
# 2. 服务器定义 (Server)
# ==========================================
class Server:
    def __init__(self, input_dim, output_dim, dataset_name):
        self.global_model = GraphCNN(
            num_layers=3,
            num_mlp_layers=2,
            input_dim=input_dim,
            hidden_dim=64,
            output_dim=output_dim,
            final_dropout=0.5,
            dataset_name=dataset_name
        ).to(device)

    def aggregate(self, local_weights_list):
        # 标准 FedAvg 聚合逻辑
        avg_weights = copy.deepcopy(local_weights_list[0])
        for key in avg_weights.keys():
            for i in range(1, len(local_weights_list)):
                avg_weights[key] += local_weights_list[i][key]
            avg_weights[key] = torch.div(avg_weights[key], len(local_weights_list))
        self.global_model.load_state_dict(avg_weights)

    def evaluate(self, test_data):
        self.global_model.eval()
        test_labels = torch.tensor([g.label for g in test_data], dtype=torch.long).to(device)
        with torch.no_grad():
            logits = self.global_model(test_data)
            preds = torch.argmax(logits, dim=1)
            correct = (preds == test_labels).sum().item()
            acc = correct / len(test_data) * 100
        return acc


# ==========================================
# 3. 命令行参数
# ==========================================
def get_args():
    parser = argparse.ArgumentParser(description="Federated Graph Learning Framework")
    parser.add_argument('--dataset', type=str, default='MUTAG')
    parser.add_argument('--num_clients', type=int, default=20)
    parser.add_argument('--num_selected', type=int, default=10)
    parser.add_argument('--malicious_ratio', type=float, default=0.0)
    parser.add_argument('--num_rounds', type=int, default=200)
    parser.add_argument('--local_epochs', type=int, default=1)
    parser.add_argument('--attack_method', type=str, default='no_attack')
    return parser.parse_args()


# ==========================================
# 4. 主流程
# ==========================================
if __name__ == "__main__":
    args = get_args()

    # 创建输出目录
    output_dir = "./output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    log_file = os.path.join(output_dir, f"{args.dataset}_training_log.csv")

    print(f"========== 启动联邦图学习 ({args.dataset}) ==========")

    # 1. 加载数据并映射标签
    graphs = get_dataset(args.dataset)
    raw_labels = [g.label for g in graphs]
    unique_labels = sorted(list(set(raw_labels)))
    label_map = {old: new for new, old in enumerate(unique_labels)}

    for g in graphs:
        g.label = label_map[g.label]
        g.node_features = g.node_features.to(device)
        g.edge_mat = g.edge_mat.to(device)

    input_dim = graphs[0].node_features.shape[1]
    output_dim = len(unique_labels)

    # 2. 切分联邦数据
    clients_data, test_data = split_federated_data(graphs, args.num_clients)

    # 3. 初始化 Server 和 Clients
    server = Server(input_dim, output_dim, args.dataset)
    num_malicious = int(args.num_clients * args.malicious_ratio)
    clients = [Client(i, clients_data[i], i < num_malicious, input_dim, output_dim, args.dataset)
               for i in range(args.num_clients)]

    # 准备保存记录的列表
    history = []

    # 4. 训练循环
    for r in range(1, args.num_rounds + 1):
        selected_clients = random.sample(clients, args.num_selected)
        global_weights = server.global_model.state_dict()

        local_weights_list, local_loss_list = [], []
        for client in selected_clients:
            weights, loss = client.train(global_weights, args.local_epochs, args.attack_method, r)
            local_weights_list.append(weights)
            local_loss_list.append(loss)

        avg_loss = sum(local_loss_list) / len(local_loss_list)
        server.aggregate(local_weights_list)

        test_acc = ""
        if r % 5 == 0 or r == 1:
            test_acc = server.evaluate(test_data)
            print(f"[Round {r:03d}] Loss: {avg_loss:.4f} | Test Acc: {test_acc:.2f}%")
        else:
            print(f"[Round {r:03d}] Loss: {avg_loss:.4f}")

        # 记录数据：[轮次, 本地平均Loss, 全局测试准确率]
        history.append([r, avg_loss, test_acc])

    # 5. 将结果写入 CSV 表格
    with open(log_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Round', 'Local_Avg_Loss', 'Test_Accuracy (%)'])
        writer.writerows(history)

    print(f"\n🎉 训练完成！结果已保存在: {log_file}")