import torch.nn as nn
import torch
import torch.nn.functional as F
import einops
from torch_geometric.datasets import TUDataset
from torch_geometric.nn import (GATConv, GCNConv, GINConv, ChebConv, global_add_pool,
                                global_sort_pool, global_max_pool, global_mean_pool)
from torch.nn.parameter import Parameter
from torch.nn import init
import math
from types import SimpleNamespace


class GateAttention(nn.Module):
    def __init__(self, in_dims, token_dim, num_heads=1):
        super().__init__()

        self.to_query = nn.Linear(in_dims, token_dim * num_heads)
        self.to_key = nn.Linear(in_dims, token_dim * num_heads)

        self.w_g = nn.Parameter(torch.randn(token_dim * num_heads, 1))
        self.scale_factor = token_dim ** -0.5
        self.Proj = nn.Linear(token_dim * num_heads, token_dim * num_heads)
        self.final = nn.Linear(token_dim * num_heads, token_dim)

    def forward(self, x):
        query = self.to_query(x)
        key = self.to_key(x)

        query = torch.nn.functional.normalize(query, dim=-1) #BxNxD
        key = torch.nn.functional.normalize(key, dim=-1) #BxNxD

        query_weight = query @ self.w_g # BxNx1 (BxNxD @ Dx1)
        A = query_weight * self.scale_factor # BxNx1

        A = torch.nn.functional.normalize(A, dim=1) # BxNx1

        G = torch.sum(A * query, dim=1) # BxD

        G = einops.repeat(
            G, "b d -> b repeat d", repeat=key.shape[1]
        ) # BxNxD

        out = self.Proj(G * key) + query #BxNxD
        out = self.final(out) # BxNxD

        return out


class MVMGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers=3, dropout=0.3, mlp_hidden=128,
                 in_dim2=None, node_num=None):
        super(MVMGNN, self).__init__()
        self.dropout = dropout
        self.node_num = node_num
        self.in_dim = in_dim

        # ----- GCN_Conv -----
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

        self.attn = GateAttention(in_dims=2 * hidden_dim, token_dim=2 * hidden_dim)

        # ----- MLP : 2 * hidden_dim -----
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1)
        )
        self.prob1 = Parameter(torch.zeros((self.node_num, self.in_dim)))
        self.prob_bias1 = Parameter(torch.empty((self.in_dim * 2, 1)))
        init.kaiming_uniform_(self.prob1, a=math.sqrt(5))
        init.kaiming_uniform_(self.prob_bias1, a=math.sqrt(5))

        self.prob2 = Parameter(torch.zeros((self.node_num, self.in_dim)))
        self.prob_bias2 = Parameter(torch.empty((self.in_dim * 2, 1)))
        init.kaiming_uniform_(self.prob2, a=math.sqrt(5))
        init.kaiming_uniform_(self.prob_bias2, a=math.sqrt(5))

        # 正则化超参数
        self.hp = SimpleNamespace(
            lamda_x_l1=1e-4,  # 特征 L1 稀疏
            lamda_e_l1=1e-4,  # 边权 L1 稀疏
            lamda_x_ent=1e-4,  # 特征 熵正则
            lamda_e_ent=1e-4,  # 边权 熵正则
        )

    def cal_probability(self, x, edge_index, edge_weight, view=1):
        """
        x: [N, D] 批图节点特征（所有图拼在一起）
        edge_index: [2, E]
        edge_weight: [E] 或 None
        view: 1 or 2
        """
        if view == 1:
            prob = self.prob1
            prob_bias = self.prob_bias1
        else:
            prob = self.prob2
            prob_bias = self.prob_bias2

        N, D = x.shape
        # 假设每张图的节点数固定为 node_num
        assert N % self.node_num == 0, "batch 内每图节点数必须等于 node_num"
        G = N // self.node_num

        x_reshaped = x.view(G, self.node_num, D)  # [G, node_num, D]
        # [node_num, D] 掩码广播到每张图
        x_prob = prob  # 如果想先做 sigmoid 再乘，可以改成 torch.sigmoid(prob)
        x_feat_prob = x_reshaped * x_prob  # [G, node_num, D]
        x_feat_prob = x_feat_prob.view(N, D)

        # 边概率
        src, dst = edge_index  # [E], [E]
        concat_prob = torch.cat(
            (x_feat_prob[src], x_feat_prob[dst]), dim=-1
        )  # [E, 2D]
        edge_prob = torch.sigmoid(concat_prob.matmul(prob_bias)).view(-1)  # [E]

        if edge_weight is None:
            edge_weight_prob = edge_prob
        else:
            edge_weight_prob = edge_weight * edge_prob

        return x_feat_prob, edge_weight_prob, x_prob, edge_prob

    def loss_probability(self, x, edge_index, edge_weight, view=1, eps=1e-6):
        hp = self.hp
        x_feat_prob, edge_weight_prob, x_prob, edge_prob = self.cal_probability(x, edge_index, edge_weight, view=view)

        x_prob = torch.sigmoid(x_prob)
        N_nodes, D = x_prob.shape
        all_num = N_nodes * D

        # 特征 L1 / 熵
        f_sum_loss = x_prob.norm(dim=-1, p=1).sum() / N_nodes
        f_entrp_loss = -torch.sum(
            x_prob * torch.log(x_prob + eps)
            + (1 - x_prob) * torch.log((1 - x_prob) + eps)
        ) / all_num

        # 边 L1 / 熵
        N_edges = edge_prob.shape[0]
        e_sum_loss = edge_prob.norm(dim=-1, p=1) / N_edges
        e_entrp_loss = -torch.sum(
            edge_prob * torch.log(edge_prob + eps)
            + (1 - edge_prob) * torch.log((1 - edge_prob) + eps)
        ) / N_edges

        loss_prob = (
                hp.lamda_x_l1 * f_sum_loss
                + hp.lamda_e_l1 * e_sum_loss
                + hp.lamda_x_ent * f_entrp_loss
                + hp.lamda_e_ent * e_entrp_loss
        )
        return x_feat_prob, edge_weight_prob, loss_prob, x_prob, edge_prob

    def forward_conv(self, x, edge_index, batch, edge_weight, convs):
        for conv in convs:
            x = conv(x, edge_index, edge_weight)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # 图级池化
        x = global_mean_pool(x, batch)
        return x

    def forward(self, data):
        """
        约定：
          data.x, data.edge_index, data.edge_weight, data.batch   -> 视图1
          data.x_2, data.edge_index_2, data.edge_weight_2         -> 视图2
        """
        batch = data.batch
        # ------- View 1 -------
        x1 = data.x
        edge_index1 = data.edge_index
        edge_weight1 = getattr(data, "edge_weight", None)
        x1, edge_weight1, loss1, _, _ = self.loss_probability(x1, edge_index1, edge_weight1, view=1)
        h1 = self.forward_conv(x1, edge_index1, batch, edge_weight1, self.convs)

        # ------- View 2 -------
        x2 = data.x_2
        edge_index2 = data.edge_index_2
        edge_weight2 = getattr(data, "edge_weight_2", None)
        x2, edge_weight2, loss2, _, _ = self.loss_probability(x2, edge_index2, edge_weight2, view=2)
        h2 = self.forward_conv(x2, edge_index2, batch, edge_weight2, self.convs)

        h1 = h1.unsqueeze(1)  # [32, 1, 128]
        h2 = h2.unsqueeze(1)  # [32, 1, 128]
        h_fused = torch.cat([h1, h2], dim=-1)  # [32, 1, 256]

        h_out = self.attn.forward(h_fused)  # [32, 1, 256]
        h = h_out.squeeze(1)  # [32, 256]

        # ------- 特征融合 + 分类 -------
        out = self.mlp(h)  # [B, 1]
        loss = (loss1 + loss2)/2

        return out, loss


class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers=3, dropout=0.3, mlp_hidden=128,
                 in_dim2=None, node_num=None,view= None):
        super(GCN, self).__init__()

        self.in_dim = in_dim
        self.node_num = node_num
        self.view = view

        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

        self.dropout = dropout

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1)
        )

    def forward(self, data, isExplain=True):

        batch, x, edge_index, edge_weight = None, None, None, None
        if self.view == 1:
            batch = data.batch
            x = data.x
            edge_index = data.edge_index
            edge_weight = getattr(data, "edge_weight", None)
        elif self.view == 2:
            batch = data.batch
            x = data.x_2
            edge_index = data.edge_index_2
            edge_weight = getattr(data, "edge_weight_2", None)

        prob_loss = 0

        for conv in self.convs:
            x = conv(x, edge_index, edge_weight)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = global_mean_pool(x, batch)
        out = self.mlp(x)

        return out, prob_loss


class GCN_view(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers=3, dropout=0.3, mlp_hidden=128,
                 in_dim2=None, node_num=None):
        super(GCN_view, self).__init__()
        self.dropout = dropout

        # ----- 图卷积堆 -----
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))


        # ----- 分类头：输入维度 = 2 * hidden_dim -----
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1)
        )

    def forward_conv(self, x, edge_index, batch, edge_weight, convs):
        for conv in convs:
            x = conv(x, edge_index, edge_weight)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # 图级池化
        x = global_mean_pool(x, batch)
        return x

    def forward(self, data):
        """
        约定：
          data.x, data.edge_index, data.edge_weight, data.batch   -> 视图1
          data.x_2, data.edge_index_2, data.edge_weight_2         -> 视图2
        """
        batch = data.batch
        # ------- View 1 -------
        x1 = data.x
        edge_index1 = data.edge_index
        edge_weight1 = getattr(data, "edge_weight", None)
        h1 = self.forward_conv(x1, edge_index1, batch, edge_weight1, self.convs)

        # ------- View 2 -------
        x2 = data.x_2
        edge_index2 = data.edge_index_2
        edge_weight2 = getattr(data, "edge_weight_2", None)
        h2 = self.forward_conv(x2, edge_index2, batch, edge_weight2, self.convs)

        # ------- 特征融合 + 分类 -------
        h = torch.cat([h1, h2], dim=-1)  # [B, 2*hidden_dim]
        out = self.mlp(h)  # [B, 1]

        return out, 0


class GCN_aview(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers=3, dropout=0.3, mlp_hidden=128,
                 in_dim2=None, node_num=None):
        super(GCN_aview, self).__init__()
        self.dropout = dropout

        # ----- 图卷积堆 -----
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

        self.attn = GateAttention(in_dims=2 * hidden_dim, token_dim=2 * hidden_dim)

        # ----- 分类头：输入维度 = 2 * hidden_dim -----
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1)
        )

    def forward_conv(self, x, edge_index, batch, edge_weight, convs):
        for conv in convs:
            x = conv(x, edge_index, edge_weight)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # 图级池化
        x = global_mean_pool(x, batch)
        return x

    def forward(self, data):
        """
        约定：
          data.x, data.edge_index, data.edge_weight, data.batch   -> 视图1
          data.x_2, data.edge_index_2, data.edge_weight_2         -> 视图2
        """
        batch = data.batch
        # ------- View 1 -------
        x1 = data.x
        edge_index1 = data.edge_index
        edge_weight1 = getattr(data, "edge_weight", None)
        h1 = self.forward_conv(x1, edge_index1, batch, edge_weight1, self.convs)

        # ------- View 2 -------
        x2 = data.x_2
        edge_index2 = data.edge_index_2
        edge_weight2 = getattr(data, "edge_weight_2", None)
        h2 = self.forward_conv(x2, edge_index2, batch, edge_weight2, self.convs)

        h1 = h1.unsqueeze(1)  # [32, 1, 128]
        h2 = h2.unsqueeze(1)  # [32, 1, 128]
        h_fused = torch.cat([h1, h2], dim=-1)  # [32, 1, 256]

        h_out = self.attn.forward(h_fused)  # [32, 1, 256] (已带全局交互)
        h = h_out.squeeze(1)  # [32, 256] 若后续需要向量

        # ------- 特征融合 + 分类 -------
        #h = torch.cat([h1, h2], dim=-1)  # [B, 2*hidden_dim]
        out = self.mlp(h)  # [B, 1]

        return out, 0


class SGCN_view(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers=3, dropout=0.3, mlp_hidden=128,
                 in_dim2=None, node_num=None):
        super(SGCN_view, self).__init__()
        self.dropout = dropout
        self.node_num = node_num
        self.in_dim = in_dim

        # ----- GCN_Conv -----
        self.conv = nn.ModuleList()
        self.conv.append(GCNConv(in_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.conv.append(GCNConv(hidden_dim, hidden_dim))

        self.attn = GateAttention(in_dims=2 * hidden_dim, token_dim=256)

        # ----- MLP : 2 * hidden_dim -----
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1)
        )

        self.prob1 = Parameter(torch.zeros((self.node_num, self.in_dim)))
        self.prob_bias1 = Parameter(torch.empty((self.in_dim * 2, 1)))
        init.kaiming_uniform_(self.prob1, a=math.sqrt(5))
        init.kaiming_uniform_(self.prob_bias1, a=math.sqrt(5))

        self.prob2 = Parameter(torch.zeros((self.node_num, self.in_dim)))
        self.prob_bias2 = Parameter(torch.empty((self.in_dim * 2, 1)))
        init.kaiming_uniform_(self.prob2, a=math.sqrt(5))
        init.kaiming_uniform_(self.prob_bias2, a=math.sqrt(5))

        # 正则化超参数
        self.hp = SimpleNamespace(
            lamda_x_l1=1e-4,  # 特征 L1 稀疏
            lamda_e_l1=1e-4,  # 边权 L1 稀疏
            lamda_x_ent=1e-4,  # 特征 熵正则
            lamda_e_ent=1e-4,  # 边权 熵正则
        )

    def forward_conv(self, x, edge_index, batch, edge_weight, convs):
        for conv in convs:
            x = conv(x, edge_index, edge_weight)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # 图级池化
        x = global_mean_pool(x, batch)
        return x

    def cal_probability(self, x, edge_index, edge_weight, view=1):
        """
        x: [N, D] 批图节点特征（所有图拼在一起）
        edge_index: [2, E]
        edge_weight: [E] 或 None
        view: 1 or 2
        """
        if view == 1:
            prob = self.prob1
            prob_bias = self.prob_bias1
        else:
            prob = self.prob2
            prob_bias = self.prob_bias2


        N, D = x.shape
        # 假设每张图的节点数固定为 node_num
        assert N % self.node_num == 0, "batch 内每图节点数必须等于 node_num"
        G = N // self.node_num

        x_reshaped = x.view(G, self.node_num, D)  # [G, node_num, D]
        # [node_num, D] 掩码广播到每张图
        x_prob = prob  # 如果想先做 sigmoid 再乘，可以改成 torch.sigmoid(prob)
        x_feat_prob = x_reshaped * x_prob  # [G, node_num, D]
        x_feat_prob = x_feat_prob.view(N, D)

        # 边概率
        src, dst = edge_index  # [E], [E]
        concat_prob = torch.cat(
            (x_feat_prob[src], x_feat_prob[dst]), dim=-1
        )  # [E, 2D]
        edge_prob = torch.sigmoid(concat_prob.matmul(prob_bias)).view(-1)  # [E]

        if edge_weight is None:
            edge_weight_prob = edge_prob
        else:
            edge_weight_prob = edge_weight * edge_prob

        return x_feat_prob, edge_weight_prob, x_prob, edge_prob

    def loss_probability(self, x, edge_index, edge_weight, view=1, eps=1e-6):
        hp = self.hp
        x_feat_prob, edge_weight_prob, x_prob, edge_prob = self.cal_probability(x, edge_index, edge_weight, view=view)

        x_prob = torch.sigmoid(x_prob)
        N_nodes, D = x_prob.shape
        all_num = N_nodes * D

        # 特征 L1 / 熵
        f_sum_loss = x_prob.norm(dim=-1, p=1).sum() / N_nodes
        f_entrp_loss = -torch.sum(
            x_prob * torch.log(x_prob + eps)
            + (1 - x_prob) * torch.log((1 - x_prob) + eps)
        ) / all_num

        # 边 L1 / 熵
        N_edges = edge_prob.shape[0]
        e_sum_loss = edge_prob.norm(dim=-1, p=1) / N_edges
        e_entrp_loss = -torch.sum(
            edge_prob * torch.log(edge_prob + eps)
            + (1 - edge_prob) * torch.log((1 - edge_prob) + eps)
        ) / N_edges

        loss_prob = (
                hp.lamda_x_l1 * f_sum_loss
                + hp.lamda_e_l1 * e_sum_loss
                + hp.lamda_x_ent * f_entrp_loss
                + hp.lamda_e_ent * e_entrp_loss
        )
        return x_feat_prob, edge_weight_prob,loss_prob, x_prob, edge_prob

    def forward(self, data):
        """
        约定：
          data.x, data.edge_index, data.edge_weight, data.batch   -> 视图1
          data.x_2, data.edge_index_2, data.edge_weight_2         -> 视图2
        """
        batch = data.batch
        # ------- View 1 -------
        x1 = data.x
        edge_index1 = data.edge_index
        edge_weight1 = getattr(data, "edge_weight", None)
        x1, edge_weight1, loss1, _, _ = self.loss_probability(x1, edge_index1, edge_weight1, view=1)
        h1 = self.forward_conv(x1, edge_index1, batch, edge_weight1, self.conv)

        # ------- View 2 -------
        x2 = data.x_2
        edge_index2 = data.edge_index_2
        edge_weight2 = getattr(data, "edge_weight_2", None)
        x2, edge_weight2, loss2, _, _ = self.loss_probability(x2, edge_index2, edge_weight2, view=2)
        h2 = self.forward_conv(x2, edge_index2, batch, edge_weight2, self.conv)

        h_fused = torch.cat([h1, h2], dim=-1)  # [32, 1, 256]

        # ------- 特征融合 + 分类 -------
        out = self.mlp(h_fused)  # [B, 1]
        loss = (loss1 + loss2)/2

        return out, loss


