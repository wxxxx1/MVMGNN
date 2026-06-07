from torch_geometric.nn import global_mean_pool
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv,SAGEConv,GCNConv
from typing import Optional

from torch import Tensor

from torch_geometric.utils import scatter
import torch_geometric.typing
from torch_geometric import is_compiling, warnings
from torch_geometric.typing import torch_scatter
from torch_geometric.utils.functions import cumsum
import numpy as np


class Model(nn.Module):
    def __init__(self, in_dim, node_num, hidden_dim, dropout, num_layers,
                 cdim = 32, ncluster = 8,
                 layer='GCN'):
        super(Model, self).__init__()

        self.dropout = dropout
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.bn = nn.BatchNorm1d(hidden_dim)
        roi = node_num
        self.poolings = nn.ModuleList(
            [global_cluster_pool(roi, hidden_dim, cdim, ncluster=ncluster, dropout=self.dropout) for _ in
             range(num_layers)])

        input_dim = in_dim
        for i in range(self.num_layers):
            if layer == 'GAT':
                self.convs.append(CustomGATLayer(input_dim, hidden_dim, 1, ))
            elif layer == 'GCN':
                self.convs.append(CustomGCNLayer(input_dim, hidden_dim))
            elif layer == 'Sage':
                self.convs.append(CustomSageLayer(input_dim, hidden_dim))

            input_dim = hidden_dim

            # post-message-passing
        self.post_mp = nn.Sequential(
            nn.Linear(cdim * num_layers * ncluster, hidden_dim), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(hidden_dim, 1))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        edge_attr, batch = data.edge_attr, data.batch
        pos = getattr(data, "pos", None)
        out = x
        cov = x[:, 1:].reshape(-1, 4)

        layer_outs = []
        for layers, poolings in zip(self.convs, self.poolings):
            out, edge_index, edge_attr, batch = layers(out, edge_index, edge_attr, batch)
            out = F.dropout(out, p=self.dropout, training=self.training)
            mean_pool, cluster_indices = poolings(out, batch, pos=pos)
            layer_outs.append(mean_pool)

        out = torch.cat(layer_outs, dim=1)
        out = self.post_mp(out)


        return out.reshape(-1), 0


class global_cluster_pool(nn.Module):
    def __init__(self,roi,hidden_dim,cdim,ncluster=14,dropout=0.5):
        super(global_cluster_pool, self).__init__()
        self.c=ncluster
        self.cluster_layers1 = nn.ModuleList()
        self.cluster_layers2 = nn.ModuleList()
        self.relu = nn.LeakyReLU()
        self.droupout=dropout
        self.r=roi
        self.linear_model=nn.Linear(self.r, self.c, bias=False)
        for i in range(self.c):
            layer1 = nn.Linear(hidden_dim, hidden_dim)
            bn1 = nn.BatchNorm1d(hidden_dim)
            layer2 = nn.Linear(hidden_dim, cdim)
            bn2 = nn.BatchNorm1d(cdim)

            torch.nn.init.kaiming_uniform_(layer1.weight, nonlinearity='leaky_relu')
            torch.nn.init.kaiming_uniform_(layer2.weight, nonlinearity='leaky_relu')

            self.cluster_layers1.append(nn.Sequential(layer1, bn1))
            self.cluster_layers2.append(nn.Sequential(layer2, bn2))

    def forward(self,x: Tensor, batch: Optional[Tensor],pos,
                     size: Optional[int] = None)-> Tensor:
        self.dim = -1 if isinstance(x, Tensor) and x.dim() == 1 else -2
        self.linear_model=self.linear_model.to(x.device)
        if pos is None:
            pos = x.new_zeros((x.size(0), self.r))  # (N, roi) 全0
        cluster_probs =nn.functional.softmax(self.linear_model(pos), dim=-1)
        _, cluster_indices = torch.max(cluster_probs, dim=-1)

        cluster_representations = []
        for i in range(self.c):
            mask = (cluster_indices == i).float().reshape(-1,1)
            cluster_features = x * mask
            cluster_out=scatter(cluster_features, batch, dim=self.dim, dim_size=size, reduce='mean')
            cluster_out=self.cluster_layers1[i](cluster_out)
            cluster_out=self.relu(cluster_out)
            cluster_out=F.dropout(cluster_out, p=self.droupout)
            cluster_out=self.cluster_layers2[i](cluster_out)
            cluster_out=self.relu(cluster_out)
            cluster_out=F.dropout(cluster_out, p=self.droupout)
            cluster_representations.append(cluster_out)

        out=torch.cat(cluster_representations,dim=1)

        return out,cluster_indices,


class CustomGATLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, edge_dim, ):
        super(CustomGATLayer, self).__init__()
        self.conv = GATConv(input_dim, hidden_dim, edge_dim=edge_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.conv.forward(x, edge_index, edge_attr)
        x = self.bn(x)
        x = self.relu(x)
        return x, edge_index, edge_attr, batch,


class CustomGCNLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(CustomGCNLayer, self).__init__()
        self.conv = GCNConv(input_dim, hidden_dim, normalize=False)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.conv.forward(x, edge_index, edge_attr)
        x = self.bn(x)
        x = self.relu(x)
        return x, edge_index, edge_attr, batch


class CustomSageLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(CustomSageLayer, self).__init__()
        self.conv = SAGEConv(input_dim, hidden_dim, )
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.conv.forward(x, edge_index)
        x = self.bn(x)
        x = self.relu(x)
        return x, edge_index, edge_attr, batch


