import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        """
        灵活的多层感知机，既可以作为普通的线性分类器，也可以作为 GCN 内部的特征提取器。
        对齐 Opt-GDBA 的标准实现。
        """
        super(MLP, self).__init__()
        self.num_layers = num_layers

        if num_layers < 1:
            raise ValueError("MLP的层数必须大于等于 1 !")

        self.linears = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        if num_layers == 1:
            # 单层退化为线性模型
            self.linears.append(nn.Linear(input_dim, output_dim))
        else:
            # 多层深度模型
            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for _ in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            # Opt-GDBA 标准：为除最后一层外的每一层添加 BatchNorm
            for _ in range(num_layers - 1):
                self.batch_norms.append(nn.LayerNorm(hidden_dim))

    def forward(self, x):
        if self.num_layers == 1:
            return self.linears[0](x)

        h = x
        for layer in range(self.num_layers - 1):
            # 顺序严格对齐 Opt-GDBA: Linear -> BatchNorm -> ReLU
            h = F.relu(self.batch_norms[layer](self.linears[layer](h)))
        # 最后一层直接输出
        return self.linears[-1](h)