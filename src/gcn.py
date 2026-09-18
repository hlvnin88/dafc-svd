"""GCN 2 lớp cho F3–F7."""
import torch
import torch.nn as nn
import torch.nn.functional as F

class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)
    def forward(self, x, edge_index):
        n = x.size(0)
        if edge_index.numel() == 0:
            edge_index = torch.arange(n, device=x.device).unsqueeze(0).repeat(2, 1)
        row, col = edge_index
        deg = torch.zeros(n, device=x.device).scatter_add_(
            0, row, torch.ones_like(row, dtype=torch.float))
        dinv = (deg + 1).pow(-0.5)
        x = x * dinv.unsqueeze(-1)
        agg = torch.zeros_like(x).index_add_(0, row, x[col])
        agg = agg * dinv.unsqueeze(-1)
        return F.relu(self.lin(agg))

class GCNEncoder(nn.Module):
    def __init__(self, in_dim, hidden=128, out_dim=256, dropout=0.3):
        super().__init__()
        self.gcn1 = GCNLayer(in_dim, hidden)
        self.gcn2 = GCNLayer(hidden, hidden)
        self.proj = nn.Sequential(nn.Linear(hidden, out_dim), nn.ReLU(),
                                   nn.Dropout(dropout))
    def forward(self, x, edge_index, batch_vec=None):
        h = self.gcn1(x, edge_index)
        h = self.gcn2(h, edge_index)
        if batch_vec is None:
            g = h.mean(dim=0, keepdim=True)
        else:
            B = int(batch_vec.max().item()) + 1
            g = torch.zeros(B, h.size(1), device=h.device)
            c = torch.zeros(B, 1, device=h.device)
            g.index_add_(0, batch_vec, h)
            c.index_add_(0, batch_vec, torch.ones_like(h[:, :1]))
            g = g / c.clamp(min=1)
        return self.proj(g)