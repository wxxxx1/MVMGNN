import torch
import einops
from torch.nn.parameter import Parameter
from torch.nn import init
import math
from types import SimpleNamespace
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch_scatter import scatter_mean, scatter_add, scatter


# ========= Layers =========
class FILConv(nn.Module):
    def __init__(self, in_feats, out_feats, norm=None, activation=None):
        super(FILConv, self).__init__()
        self._in_src_feats = in_feats
        self._out_feats = out_feats
        self.norm = norm
        self.activation = activation
        self.fc_neigh = nn.Linear(self._in_src_feats, out_feats, bias=False)
        self.fc_self = nn.Linear(self._in_src_feats, out_feats)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_uniform_(self.fc_neigh.weight, gain=gain)
        nn.init.xavier_uniform_(self.fc_self.weight, gain=gain)
        if self.fc_self.bias is not None:
            nn.init.zeros_(self.fc_self.bias)

    def forward(self, x, edge_index, edge_weight=None):
        num_nodes = x.size(0)
        src, dst = edge_index[0], edge_index[1]
        h_src = self.fc_neigh(x[src])
        if edge_weight is not None:
            h_src = h_src * edge_weight.unsqueeze(-1)
        h_neigh = scatter_mean(h_src, dst, dim=0, dim_size=num_nodes)
        h_self = self.fc_self(x)
        out = h_self + h_neigh
        if self.activation is not None:
            out = self.activation(out)
        if self.norm is not None:
            out = self.norm(out)
        return out

class GINConv(nn.Module):
    def __init__(self, apply_func, neighbor_pooling_type, init_eps, learn_eps):
        super(GINConv, self).__init__()
        self.apply_func = apply_func
        self._init_eps = init_eps
        if learn_eps:
            self.eps = nn.Parameter(torch.Tensor([init_eps]))
        else:
            self.register_buffer('eps', torch.Tensor([init_eps]))
        self.neighbor_pooling_type = neighbor_pooling_type

    def forward(self, x, edge_index):
        num_nodes = x.size(0)
        src, dst = edge_index[0], edge_index[1]
        if self.neighbor_pooling_type == 'sum':
            h_neigh = scatter_add(x[src], dst, dim=0, dim_size=num_nodes)
        elif self.neighbor_pooling_type == 'mean':
            h_neigh = scatter_mean(x[src], dst, dim=0, dim_size=num_nodes)
        elif self.neighbor_pooling_type == 'max':
            h_neigh = scatter(x[src], dst, dim=0, dim_size=num_nodes, reduce='max')
        else:
            raise NotImplementedError
        combined = (1 + self.eps) * x + h_neigh
        return self.apply_func(combined)

class SparseMHA(nn.Module):
    """Graph-aware MHA（沿边消息，row-wise softmax，融合 edge_weight）"""
    def __init__(self, hidden_size=80, num_heads=8):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scaling = self.head_dim ** -0.5
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, edge_index, edge_weight=None):
        N = x.size(0)
        src_nodes, dst_nodes = edge_index[0], edge_index[1]
        q = self.q_proj(x).view(N, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(N, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(N, self.num_heads, self.head_dim)
        outs = []
        for h in range(self.num_heads):
            q_h = q[:, h, :]
            k_h = k[:, h, :]
            v_h = v[:, h, :]
            scores = (q_h[src_nodes] * k_h[dst_nodes]).sum(dim=-1) * self.scaling
            if edge_weight is not None:
                scores = scores * edge_weight
            max_per_src = scatter(scores, src_nodes, dim=0, dim_size=N, reduce='max')
            scores_exp = torch.exp(scores - max_per_src[src_nodes])
            denom = scatter_add(scores_exp, src_nodes, dim=0, dim_size=N)
            alpha = scores_exp / (denom[src_nodes] + 1e-10)
            out_h = scatter_add(v_h[dst_nodes] * alpha.unsqueeze(-1), src_nodes, dim=0, dim_size=N)
            outs.append(out_h)
        out = torch.cat(outs, dim=-1)
        return self.out_proj(out)

class GTLayer(nn.Module):
    def __init__(self, hidden_size=80, num_heads=8):
        super().__init__()
        self.MHA = SparseMHA(hidden_size=hidden_size, num_heads=num_heads)
        self.batchnorm1 = nn.BatchNorm1d(hidden_size)
        self.batchnorm2 = nn.BatchNorm1d(hidden_size)
        self.FFN1 = nn.Linear(hidden_size, hidden_size * 2)
        self.FFN2 = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, h, edge_index, edge_weight=None):
        h1 = h
        h = self.MHA.forward(h, edge_index, edge_weight)
        h = self.batchnorm1(h + h1)
        h2 = h
        h = self.FFN2(F.relu(self.FFN1(h)))
        return self.batchnorm2(h + h2)


# ========= Blocks =========
class FILmodule(nn.Module):
    def __init__(self, in_dim, n_hidden, out_dim, n_layers, activation):
        super(FILmodule, self).__init__()
        self.layers = nn.ModuleList()
        self.activation = activation
        self.layers.append(FILConv(in_dim, n_hidden))
        for _ in range(n_layers - 1):
            self.layers.append(FILConv(n_hidden, n_hidden))
        self.layers.append(FILConv(n_hidden, out_dim))

    def forward(self, x, edge_index, edge_weight):
        h = x
        for l, layer in enumerate(self.layers):
            h = layer(h, edge_index, edge_weight)
            if l != len(self.layers) - 1:
                h = self.activation(h)
        return h

class ApplyNodeFunc(nn.Module):
    def __init__(self, mlp):
        super(ApplyNodeFunc, self).__init__()
        self.mlp = mlp
        self.bn = nn.BatchNorm1d(self.mlp.output_dim)

    def forward(self, h):
        h = self.mlp(h)
        h = self.bn(h)
        h = F.relu(h)
        return h

class MLP(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.linear_or_not = True
        self.num_layers = num_layers
        self.output_dim = output_dim
        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()
            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for _ in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))
            for _ in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

    def forward(self, x):
        if self.linear_or_not:
            return self.linear(x)
        h = x
        for i in range(self.num_layers - 1):
            h = h.to(torch.float32)
            h = F.relu(self.batch_norms[i](self.linears[i](h)))
        return self.linears[-1](h)

class GraphPooling(nn.Module):
    def __init__(self, pooling_type):
        super().__init__()
        self.pooling_type = pooling_type

    def forward(self, node_features, num_nodes_per_graph):
        idx = torch.cat([torch.full((n,), i, dtype=torch.long, device=node_features.device)
                         for i, n in enumerate(num_nodes_per_graph)], dim=0)
        num_graphs = len(num_nodes_per_graph)
        if self.pooling_type == 'sum':
            return scatter_add(node_features, idx, dim=0, dim_size=num_graphs)
        elif self.pooling_type == 'mean':
            return scatter_mean(node_features, idx, dim=0, dim_size=num_graphs)
        elif self.pooling_type == 'max':
            return scatter(node_features, idx, dim=0, dim_size=num_graphs, reduce='max')
        else:
            raise NotImplementedError



# ========= Model =========
class DMGNN_old(nn.Module):
    """
    DMGNN model (PyTorch only, no DGL)
    """

    def __init__(self, num_layers, num_mlp_layers, input_dim, hidden_dim,
                 output_dim, final_dropout, learn_eps, graph_pooling_type,
                 neighbor_pooling_type):
        super(DMGNN_old, self).__init__()
        self.num_layers = num_layers
        self.learn_eps = learn_eps

        self.FIL = FILmodule(input_dim, 32, hidden_dim, 2, F.relu)  # FIL module maps to hidden_dim

        self.ginlayers = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(self.num_layers - 1):
            if layer == 0:
                mlp = MLP(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim)  # FIL output is hidden_dim
            else:
                mlp = MLP(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim)

            self.ginlayers.append(
                GINConv(ApplyNodeFunc(mlp), neighbor_pooling_type, 0, self.learn_eps))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        self.linears_prediction = torch.nn.ModuleList()
        self.linears_prediction_ts = torch.nn.ModuleList()
        for layer in range(num_layers):
            if layer == 0:
                self.linears_prediction.append(
                    nn.Linear(hidden_dim, output_dim))  # FIL output
                self.linears_prediction_ts.append(
                    nn.Linear(hidden_dim, output_dim))  # GT output (initial h_ts)
            else:
                self.linears_prediction.append(
                    nn.Linear(hidden_dim, output_dim))
                self.linears_prediction_ts.append(
                    nn.Linear(hidden_dim, output_dim))

        self.drop = nn.Dropout(final_dropout)

        num_heads = 4
        num_layers_gt = num_layers + 1  # +1 for the initial GT layer
        self.gt_layers = nn.ModuleList(
            [GTLayer(hidden_dim, num_heads) for _ in range(num_layers_gt)]  # GT layers operate on hidden_dim
        )

        self.pool = GraphPooling(graph_pooling_type)  # Our custom pooling

    def forward(self, all_view_data, num_nodes_per_graph):
        """
        x: (N, input_dim) - All nodes features in the batch
        edge_index: (2, E) - All edges in the batch
        edge_weight: (E,) - All edge weights in the batch
        num_nodes_per_graph: (B,) - List of node counts for each graph in the batch
        """
        x, edge_index, edge_weight = all_view_data[1]
        # FIL Module
        h = self.FIL.forward(x, edge_index, edge_weight)

        # Graph Transformer (GT) Layer
        # A is implicitly represented by edge_index
        h_ts = self.gt_layers[0](h, edge_index)

        hidden_rep = [h]
        hidden_h_ts_rep = [h_ts]

        # GIN Layers and subsequent GT Layers
        for i in range(self.num_layers - 1):
            h = self.ginlayers[i](h, edge_index)  # GINConv_NoDGL
            h = self.batch_norms[i](h)
            h = F.relu(h)

            h_ts = self.gt_layers[i + 1](h, edge_index)  # GTLayer_NoDGL
            hidden_rep.append(h)
            hidden_h_ts_rep.append(h_ts)

        score_over_layer = 0

        # Readout (Pooling and Prediction)
        for i, h_current in enumerate(hidden_rep):
            pooled_h = self.pool.forward(h_current, num_nodes_per_graph)
            pooled_h_ts = self.pool.forward(hidden_h_ts_rep[i], num_nodes_per_graph)

            pooled_h = pooled_h.to(torch.float32)
            pooled_h_ts = pooled_h_ts.to(torch.float32)

            score_over_layer += self.drop(self.linears_prediction[i](pooled_h))
            score_over_layer += self.drop(self.linears_prediction_ts[i](pooled_h_ts))

        return score_over_layer



class DMGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers, dropout, mlp_hidden,
                 learn_eps = 1e-4, num_mlp_layers = 2,
                 neighbor_pooling_type = "mean", graph_pooling_type = "max",
                 in_dim2 = None, node_num = None, view = None):
        super(DMGNN, self).__init__()
        self.num_layers = num_layers
        self.learn_eps = learn_eps
        self.node_num = node_num
        self.in_dim = in_dim

        self.FIL = FILmodule(in_dim, hidden_dim, hidden_dim, 4, F.relu)

        self.ginlayers = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(num_layers - 1):
            mlp = MLP(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim)
            self.ginlayers.append(
                GINConv(ApplyNodeFunc(mlp), neighbor_pooling_type, 0, self.learn_eps))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        self.linears_prediction = torch.nn.ModuleList()
        self.linears_prediction_ts = torch.nn.ModuleList()
        for layer in range(num_layers):
            self.linears_prediction.append(nn.Linear(hidden_dim, 1))
            self.linears_prediction_ts.append(nn.Linear(hidden_dim, 1))

        self.drop = nn.Dropout(dropout)
        num_heads = 4
        num_layers_gt = num_layers + 1
        self.gt_layers = nn.ModuleList([GTLayer(hidden_dim, num_heads) for _ in range(num_layers_gt)])

        self.pool = GraphPooling(graph_pooling_type)

        self.prob = Parameter(torch.zeros((self.node_num, self.in_dim)))
        self.prob_bias = Parameter(torch.empty((self.in_dim * 2, 1)))
        init.kaiming_uniform_(self.prob_bias, a=math.sqrt(5))
        self.edge_prob = Parameter(torch.empty((self.node_num, self.node_num)))
        init.kaiming_uniform_(self.prob, a=math.sqrt(5))
        init.kaiming_uniform_(self.edge_prob, a=math.sqrt(5))

        self.hp = SimpleNamespace(
            lamda_x_l1=1e-4,  # 特征 L1 稀疏
            lamda_e_l1=1e-4,  # 边权 L1 稀疏
            lamda_x_ent=1e-4,  # 特征 熵正则
            lamda_e_ent=1e-4)

    def cal_probability(self, x, edge_index, edge_weight):
        N, D = x.shape
        x = x.reshape(N // self.node_num, self.node_num, D)
        x_prob = self.prob  # torch.sigmoid(self.prob)
        x_feat_prob = x * x_prob
        x_feat_prob = x_feat_prob.reshape(N, D)

        conat_prob = torch.cat((x_feat_prob[edge_index[0]], x_feat_prob[edge_index[1]]), -1)
        edge_prob = torch.sigmoid(conat_prob.matmul(self.prob_bias)).view(-1)
        if edge_weight is None:
            edge_weight_prob = edge_prob
        else:
            edge_weight_prob = edge_weight * edge_prob
        return x_feat_prob, edge_weight_prob, x_prob, edge_prob

    def loss_probability(self, x, edge_index, edge_weight, hp, eps=1e-6):
        x_feat_prob, edge_weight_prob, x_prob, edge_prob = self.cal_probability(x, edge_index, edge_weight)

        x_prob = torch.sigmoid(x_prob)

        N, D = x_prob.shape
        all_num = (N * D)
        # f_sum_loss = torch.sum(x_prob)/all_num
        f_sum_loss = x_prob.norm(dim=-1, p=1).sum() / N
        f_entrp_loss = -torch.sum(
            x_prob * torch.log(x_prob + eps) + (1 - x_prob) * torch.log((1 - x_prob) + eps)) / all_num

        N = edge_prob.shape[0]
        all_num = N
        # e_sum_loss = torch.sum(edge_prob)/all_num
        e_sum_loss = edge_prob.norm(dim=-1, p=1) / N
        e_entrp_loss = -torch.sum(
            edge_prob * torch.log(edge_prob + eps) + (1 - edge_prob) * torch.log((1 - edge_prob) + eps)) / all_num

        # sum_loss = (f_sum_loss+e_sum_loss+f_entrp_loss+e_entrp_loss)/4
        loss_prob = hp.lamda_x_l1 * f_sum_loss + hp.lamda_e_l1 * e_sum_loss + hp.lamda_x_ent * f_entrp_loss + hp.lamda_e_ent * e_entrp_loss

        return x_feat_prob, edge_weight_prob, loss_prob, x_prob, edge_prob

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        edge_weight = getattr(data, "edge_weight", None)

        batch = data.batch
        batch_size = batch.max().item() + 1
        num_nodes_per_graph = torch.full((batch_size,), self.node_num,
                                         dtype=torch.long,
                                         device=x.device)
        x, edge_weight, loss1, _, _ = self.loss_probability(x, edge_index, edge_weight, self.hp)

        h = self.FIL.forward(x, edge_index, edge_weight)

        h_ts = self.gt_layers[0](h, edge_index)
        hidden_rep = [h]
        hidden_h_ts_rep = [h_ts]

        for i in range(self.num_layers - 1):
            h = self.ginlayers[i](h, edge_index)
            h = self.batch_norms[i](h)
            h = F.relu(h)

            h_ts = self.gt_layers[i + 1](h, edge_index)
            hidden_rep.append(h)
            hidden_h_ts_rep.append(h_ts)

        score_over_layer = 0
        for i, h_current in enumerate(hidden_rep):
            pooled_h = self.pool.forward(h_current, num_nodes_per_graph)
            pooled_h_ts = self.pool.forward(hidden_h_ts_rep[i], num_nodes_per_graph)

            pooled_h = pooled_h.to(torch.float32)
            pooled_h_ts = pooled_h_ts.to(torch.float32)

            score_over_layer += self.drop(self.linears_prediction[i](pooled_h))
            score_over_layer += self.drop(self.linears_prediction_ts[i](pooled_h_ts))

        return score_over_layer
