import torch
import torch.nn as nn
import torch.nn.functional as F
from mlp import MLP


class GraphCNN(nn.Module):
    def __init__(self, num_layers, num_mlp_layers, input_dim, hidden_dim, output_dim, final_dropout=0.5,
                 dataset_name='MUTAG'):
        """
        :param dataset_name: 自动配置池化策略的依据
        """
        super(GraphCNN, self).__init__()
        self.num_layers = num_layers
        self.final_dropout = final_dropout
        self.dataset_name = dataset_name

        # ==========================================
        # 【FGL 核心优化 1】：自适应 Pooling 策略
        # ==========================================
        if self.dataset_name in ['MUTAG']:
            self.neighbor_pooling = "sum"
            self.graph_pooling = "sum"
        elif self.dataset_name in ['COLLAB', 'REDDIT-MULTI-5K']:
            # [社交网络图]：邻居防爆炸用 average，但【全图必须用 sum】以保留规模特征！
            self.neighbor_pooling = "average"
            self.graph_pooling = "sum"
        elif self.dataset_name in ['DD', 'PROTEINS']:
            self.neighbor_pooling = "average"
            self.graph_pooling = "average"
        else:
            self.neighbor_pooling = "sum"
            self.graph_pooling = "sum"

        self.mlps = torch.nn.ModuleList()
        # 【FGL 核心优化 2】：替换 BatchNorm 为 LayerNorm
        self.layer_norms = torch.nn.ModuleList()

        for layer in range(self.num_layers):
            if layer == 0:
                self.mlps.append(MLP(num_mlp_layers, input_dim, hidden_dim, hidden_dim))
            else:
                self.mlps.append(MLP(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim))

            # 使用 LayerNorm，完美规避联邦学习中 Client 方差不一致导致的模型崩溃
            self.layer_norms.append(nn.LayerNorm(hidden_dim))

        self.linears_prediction = torch.nn.ModuleList()
        for layer in range(self.num_layers + 1):
            in_dim = input_dim if layer == 0 else hidden_dim
            self.linears_prediction.append(nn.Linear(in_dim, output_dim))

    def forward(self, graph_list, return_embeds=False):
        logits = []
        embeds = []

        for g in graph_list:
            x = g.node_features

            # 强制自环保护 (Self-loop)
            adj = g.edge_mat + torch.eye(g.num_nodes, device=g.edge_mat.device)

            if self.neighbor_pooling == "average":
                degree = torch.sum(adj, dim=1, keepdim=True)
                adj = adj / torch.clamp(degree, min=1e-8)

            hidden_rep = [x]
            h = x

            for layer in range(self.num_layers):
                pooled_h = torch.matmul(adj, h)

                h = self.mlps[layer](pooled_h)
                # 应用 LayerNorm
                h = self.layer_norms[layer](h)
                h = F.relu(h)
                hidden_rep.append(h)

            score_over_layer = 0
            graph_embed_list = []

            for layer, h_layer in enumerate(hidden_rep):

                # 此时如果是 COLLAB，这里会执行 sum，将数千个节点的特征累加
                # 但因为经过了LayerNorm，数值不会像之前那样发生灾难性溢出
                if self.graph_pooling == "average":
                    pooled_graph = torch.mean(h_layer, dim=0, keepdim=True)
                else:
                    pooled_graph = torch.sum(h_layer, dim=0, keepdim=True)

                score_over_layer += F.dropout(
                    self.linears_prediction[layer](pooled_graph),
                    p=self.final_dropout,
                    training=self.training
                )
                graph_embed_list.append(pooled_graph)

            graph_embed = torch.cat(graph_embed_list, dim=1)
            logits.append(score_over_layer)
            embeds.append(graph_embed)

        logits = torch.cat(logits, dim=0)
        embeds = torch.cat(embeds, dim=0)

        if return_embeds:
            return logits, embeds
        return logits