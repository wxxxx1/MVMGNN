import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj
from torch_geometric.nn import global_mean_pool


def default_get_U(evidence: torch.Tensor, num_classes: int):
    """
    evidence: (B,C) >=0
    返回 b: (B,C), u: (B,)
    这里给一个“能跑通”的版本：Dirichlet alpha=evidence+1
    u = C / sum(alpha)  (常见不确定性定义之一)
    b = evidence / sum(alpha)  (简化的 belief)
    你有自己的 get_U 就替换掉。
    """
    alpha = evidence + 1.0
    S = torch.sum(alpha, dim=-1, keepdim=True)  # (B,1)
    u = (num_classes / (S.squeeze(-1) + 1e-8))  # (B,)
    b = evidence / (S + 1e-8)                   # (B,C)
    return b, u


class IntraViewBlock(nn.Module):
    """
    单视图：把 PyG 的 edge_index/edge_weight -> dense A，然后做 A(XW) + pool + evidence
    可选：插入你原来的 Structural_Enhancement(A, X, k) 做结构增强。
    """
    def __init__(self, in_dim: int, hidden_dim: int, num_classes: int, dropout: float):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.LeakyReLU(0.2)
        self.cls = nn.Sequential(nn.Linear(hidden_dim, num_classes), nn.Softplus())
        self.num_classes = num_classes

    def forward(self, x, edge_index, edge_weight, batch,
                structural_enhance_fn=None, get_u_fn=default_get_U,
                max_t: int = 8, step: int = 5):
        """
        x: (N,F)
        edge_index: (2,E)
        edge_weight: (E,)
        batch: (N,) or None
        输出：
          feat_g: (B,H) 图级 embedding
          evidence: (B,C)
          b: (B,C), u: (B,)
          A_used: (B,Nmax,Nmax) dense（调试用，可能很大）
        """
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)

        # dense adjacency: (B, Nmax, Nmax)
        A = to_dense_adj(edge_index, batch=batch, edge_attr=edge_weight)  # float
        B, Nmax, _ = A.shape
        eye = torch.eye(Nmax, device=x.device, dtype=A.dtype).unsqueeze(0)  # (1,N,N)

        # dense node features: (B, Nmax, F)
        # 需要把拼接的 x 按 batch 填充到 dense
        # 简单做法：用 mask scatter（不引入额外依赖）
        Fdim = x.size(-1)
        X = x.new_zeros((B, Nmax, Fdim))
        # 每个 batch 内节点的局部索引
        # 统计每张图节点数
        counts = torch.bincount(batch, minlength=B)  # (B,)
        # 为每个节点分配其在图内的顺序号
        local_idx = torch.zeros_like(batch)
        start = 0
        for bi, c in enumerate(counts.tolist()):
            if c == 0:
                continue
            local_idx[start:start+c] = torch.arange(c, device=x.device)
            start += c
        X[batch, local_idx] = x

        # 迭代结构增强（可选），用不确定性 u 作停止标准
        best_u = x.new_full((B,), 1e9)
        best_feat = x.new_zeros((B, self.proj.out_features))
        best_evid = x.new_zeros((B, self.num_classes))
        best_b = x.new_zeros((B, self.num_classes))
        best_A = A.clone()

        # 预先算一次 XW（每次 A 变，不用重复 XW 的线性）
        XW = self.proj(X)  # (B,N,H)

        for t in range(1, max_t + 1):
            if structural_enhance_fn is not None:
                # 注意：你的 Structural_Enhancement 原来是 per-sample，
                # 这里先逐 batch 调用，保证接口兼容
                A_t_list = []
                for i in range(B):
                    A_i = structural_enhance_fn(best_A[i], X[i], step * t)
                    A_t_list.append(A_i)
                A_t = torch.stack(A_t_list, dim=0)
            else:
                A_t = best_A

            A_t = 0.5 * A_t + eye

            H = torch.bmm(A_t, XW)              # (B,N,H)
            H = self.dropout(H)
            H = self.act(H)

            # 图级池化：对 dense 的节点维度均值
            feat_g = H.mean(dim=1)              # (B,H)
            evid = self.cls(feat_g)             # (B,C)
            b, u = get_u_fn(evid, self.num_classes)

            improved = u < best_u
            if improved.any():
                idx = improved.nonzero(as_tuple=False).squeeze(-1)
                best_u[idx] = u[idx]
                best_feat[idx] = feat_g[idx]
                best_evid[idx] = evid[idx]
                best_b[idx] = b[idx]
                best_A[idx] = A_t[idx]

            # 如果全都不再改善就停止
            if not improved.any():
                break

        return best_feat, best_evid, best_b, best_u, best_A


class MVGNN(nn.Module):
    """
    双视图 DMGNN：
      - view1 Intra
      - view2 Intra
      - 2x2 inter_gcn 融合
      - 输出融合 evidence
    """
    def __init__(self, in_dim, node_num, hidden_dim, num_layers,
                 view=2, dropout=0.5, mlp_hidden=None,
                 num_classes=2,
                 use_uncertainty_weight=True,in_dim2= None):
        super().__init__()
        assert view == 2
        self.view = view
        self.num_classes = num_classes
        self.use_uncertainty_weight = use_uncertainty_weight

        self.intra1 = IntraViewBlock(in_dim, hidden_dim, num_classes, dropout)
        self.intra2 = IntraViewBlock(in_dim, hidden_dim, num_classes, dropout)

        # 视图融合矩阵 2x2
        self.inter_gcn = nn.Parameter(torch.empty(2, 2))
        nn.init.xavier_uniform_(self.inter_gcn)


        self.fuse_logit = nn.Linear(hidden_dim, 1)  # ✅ 输出 logit 给 BCEWithLogitsLoss 用

        self.pro_loss_weight = 1.0  # 你也可以从 params 传进来

    def forward(self, data, epoch=None,
                structural_enhance_fn=None,
                get_u_fn=default_get_U):
        x1 = data.x
        ei1 = data.edge_index
        ew1 = getattr(data, "edge_weight", None)
        if ew1 is None and hasattr(data, "edge_attr"):
            ew1 = data.edge_attr.view(-1)
        if ew1 is None:
            ew1 = x1.new_ones(ei1.size(1))

        x2 = data.x_2
        ei2 = data.edge_index_2
        ew2 = getattr(data, "edge_weight_2", None)
        if ew2 is None and hasattr(data, "edge_attr_2"):
            ew2 = data.edge_attr_2.view(-1)
        if ew2 is None:
            ew2 = x2.new_ones(ei2.size(1))

        batch = getattr(data, "batch", None)

        # 单视图
        f1, e1, b1, u1, A1 = self.intra1.forward(
            x1, ei1, ew1, batch,
            structural_enhance_fn=structural_enhance_fn,
            get_u_fn=get_u_fn
        )
        f2, e2, b2, u2, A2 = self.intra2.forward(
            x2, ei2, ew2, batch,
            structural_enhance_fn=structural_enhance_fn,
            get_u_fn=get_u_fn
        )

        feats = torch.stack([f1, f2], dim=1)  # (B,2,H)

        if self.use_uncertainty_weight:
            b_stack = torch.stack([b1, b2], dim=1)  # (B,2,C)
            u_stack = torch.stack([u1, u2], dim=1)  # (B,2)
            var_b = torch.var(b_stack, dim=-1, unbiased=False)  # (B,2)
            w = torch.exp(var_b) / (u_stack + 1e-8)  # (B,2)
            feats_u = feats * w.unsqueeze(-1)
        else:
            feats_u = feats

        fused_views = torch.einsum("vk,bkh->bvh", self.inter_gcn, feats_u)
        fused_feat = fused_views.sum(dim=1)  # (B,H)

        logits = self.fuse_logit(fused_feat)  # (B,1)

        pro_loss = (u1.mean() + u2.mean()) * getattr(self, "pro_loss_weight", 1.0)

        return logits, pro_loss

