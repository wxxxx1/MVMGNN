import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool


def edge_index_to_sparse_adj(edge_index: torch.Tensor,
                             edge_weight: torch.Tensor,
                             num_nodes: int,
                             device=None) -> torch.Tensor:
    """
    edge_index: (2, E)
    edge_weight: (E,) or None
    returns: sparse COO adj (N, N)
    """
    if device is None:
        device = edge_index.device
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), device=device, dtype=torch.float32)
    else:
        edge_weight = edge_weight.to(device=device, dtype=torch.float32).view(-1)

    adj = torch.sparse_coo_tensor(
        indices=edge_index.to(device),
        values=edge_weight,
        size=(num_nodes, num_nodes),
        device=device,
        dtype=torch.float32
    )
    return adj.coalesce()


class meatGCN(nn.Module):
    """
    ✅ Interface compatible with your Trainer:
        out, pro_loss = model(batch)

    Expected batch fields (from your dataloader):
      View1: batch.x, batch.edge_index, batch.edge_weight (optional)
      View2: batch.x_2, batch.edge_index_2, batch.edge_weight_2 (optional)
      PyG:   batch.batch  (node -> graph id)

    Output:
      out: (B, 1) logits
      pro_loss: scalar tensor
    """
    def __init__(self,
                 in_dim: int,
                 in_dim2: int,
                 node_num: int,
                 hidden_dim: int,
                 num_layers: int,   # kept for compatibility; currently not used (Meta_GCN is 2-layer)
                 mlp_hidden: int,
                 dropout: float = 0.5):

        super().__init__()
        self.in_dim = in_dim
        self.in_dim2 = in_dim2
        self.node_num = node_num
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        # Two encoders for two views
        self.enc1 = Meta_GCN(nfeat=in_dim,  nhid=hidden_dim, nout=hidden_dim, dropout=dropout)
        self.enc2 = Meta_GCN(nfeat=in_dim2, nhid=hidden_dim, nout=hidden_dim, dropout=dropout)

        # Fusion + classifier
        # You can change fuse="concat"/"mean"/"sum" later; concat tends to be stronger
        self.fuse = "concat"
        cls_in = hidden_dim * 2 if self.fuse == "concat" else hidden_dim

        if mlp_hidden is None:
            mlp_hidden = hidden_dim

        self.classifier = nn.Sequential(
            nn.Linear(cls_in, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1),
        )

    def forward(self, batch):
        # -------- view 1 --------
        x1 = batch.x                              # (N, Fin1)
        edge_index1 = batch.edge_index            # (2, E1)
        edge_weight1 = getattr(batch, "edge_weight", None)
        b = batch.batch                           # (N,)

        N = x1.size(0)
        adj1 = edge_index_to_sparse_adj(edge_index1, edge_weight1, N, device=x1.device)
        h1 = self.enc1.forward(x1, adj1)                  # (N, H)

        # -------- view 2 --------
        x2 = batch.x_2
        edge_index2 = batch.edge_index_2
        edge_weight2 = getattr(batch, "edge_weight_2", None)

        adj2 = edge_index_to_sparse_adj(edge_index2, edge_weight2, N, device=x1.device)
        h2 = self.enc2.forward(x2, adj2)                  # (N, H)

        # -------- fuse node embeddings --------
        h1 = F.dropout(h1, p=self.dropout, training=self.training)
        h2 = F.dropout(h2, p=self.dropout, training=self.training)

        if self.fuse == "sum":
            h = h1 + h2
        elif self.fuse == "mean":
            h = (h1 + h2) * 0.5
        else:  # concat
            h = torch.cat([h1, h2], dim=-1)       # (N, 2H)

        # -------- graph pooling -> graph embedding --------
        g = global_mean_pool(h, b)                # (B, 2H) or (B, H)

        # -------- graph classification --------
        out = self.classifier(g)                  # (B, 1)

        # keep your trainer API
        pro_loss = out.new_zeros(())              # scalar tensor 0.0
        return out, pro_loss



class Meta_GCN(nn.Module):
    """
    Two-layer GCN.
    Note: we will use it as an *encoder* to produce node embeddings (N, hidden_dim),
    then do graph pooling and graph classification outside.
    """
    def __init__(self, nfeat, nhid, nout, dropout):
        super(Meta_GCN, self).__init__()
        self.gc1 = Meta_GraphConvolution(nfeat, nhid)
        self.gc2 = Meta_GraphConvolution(nhid, nout)
        self.dropout = dropout

    def forward(self, x, adj, vars=None):
        # vars: [w1, b1, w2, b2]
        if vars is not None:
            v = [vars[i:i + 2] for i in range(0, len(vars), 2)]
            x = F.relu(self.gc1.forward(x, adj, v[0]))
            x = F.dropout(x, self.dropout, training=self.training)
            x = self.gc2.forward(x, adj, v[1])
        else:
            x = F.relu(self.gc1.forward(x, adj, None))
            x = F.dropout(x, self.dropout, training=self.training)
            x = self.gc2.forward(x, adj, None)
        return x  # (N, nout)


class Meta_GCN_yuanshi(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(Meta_GCN_yuanshi, self).__init__()

        self.gc1 = Meta_GraphConvolution(nfeat, nhid)
        self.gc2 = Meta_GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj, vars=None):  # vars: [w1, b1, w2, b2]
        if vars is not None:
            v = [vars[i:i + 2] for i in range(0, len(vars), 2)]  # v: [[w1, b1], [w2, b2]]
            x = F.relu(self.gc1.forward(x, adj, v[0]))
            x = F.dropout(x, self.dropout, training=self.training)
            x = self.gc2.forward(x, adj, v[1])
        else:
            x = F.relu(self.gc1.forward(x, adj, None))
            x = F.dropout(x, self.dropout, training=self.training)
            x = self.gc2.forward(x, adj, None)
        return x


class Meta_GraphConvolution(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(Meta_GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.v = nn.ParameterList()

        weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

        torch.nn.init.kaiming_normal_(weight, mode='fan_in', nonlinearity='relu')
        if bias is not None:
            torch.nn.init.constant_(bias, 0.0)
        self.v.append(weight)
        self.v.append(bias)

    def forward(self, input, adj, v=None):
        if v is None:
            v = self.v
        support = torch.mm(input, v[0])
        output = torch.spmm(adj, support)
        if v[1] is not None:
            return output + v[1]
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'


class Encoder(nn.Module):
    def __init__(self, nfeat, nhid, npre):
        super(Encoder, self).__init__()

        self.linear1 = nn.Linear(nfeat, nhid)
        self.linear2 = nn.Linear(nhid, npre)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x


class Decoder(nn.Module):
    def __init__(self, npre, nhid, nout):
        super(Decoder, self).__init__()

        self.linear1 = nn.Linear(npre, nhid)
        self.linear2 = nn.Linear(nhid, nout)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = torch.sigmoid(self.linear2(x))
        return x


class AutoEncoder(nn.Module):
    def __init__(self, nfeat, nhid, npre, nout):
        super(AutoEncoder, self).__init__()

        self.encoder = Encoder(nfeat, nhid, npre)
        self.decoder = Decoder(npre, nhid, nout)

    def forward(self, x):
        low_feat = self.encoder.forward(x)
        high_feat = self.decoder.forward(low_feat)
        return low_feat, high_feat


class SE(nn.Module):
    def __init__(self, nfeat, ratio):
        super(SE, self).__init__()

        self.linear1 = nn.Linear(nfeat, nfeat // ratio)
        self.linear2 = nn.Linear(nfeat // ratio, nfeat)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = torch.sigmoid(self.linear2(x))
        return x