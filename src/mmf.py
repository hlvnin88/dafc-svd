"""MMAF — Multi-Metric Adaptive Fusion."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import MMAF_FUSION_DIM, MMAF_ATTN_HIDDEN, MMAF_CLS_HIDDEN, MMAF_DROPOUT

class MMAF(nn.Module):
    def __init__(self, in_dims, fusion_dim=MMAF_FUSION_DIM,
                 attn_hidden=MMAF_ATTN_HIDDEN, cls_hidden=MMAF_CLS_HIDDEN,
                 dropout=MMAF_DROPOUT):
        super().__init__()
        self.projections = nn.ModuleList([
            nn.Sequential(nn.Linear(d, fusion_dim), nn.ReLU(), nn.Dropout(dropout))
            for d in in_dims])
        self.attn = nn.Sequential(nn.Linear(fusion_dim, attn_hidden),
                                   nn.Tanh(), nn.Linear(attn_hidden, 1))
        self.cls = nn.Sequential(nn.Linear(fusion_dim, cls_hidden), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(cls_hidden, 1),
                                  nn.Sigmoid())
    def forward(self, x_list):
        h = [p(x) for p, x in zip(self.projections, x_list)]
        stacked = torch.stack(h, dim=1)
        scores = self.attn(stacked).squeeze(-1)
        alpha = F.softmax(scores, dim=1).unsqueeze(-1)
        fused = (stacked * alpha).sum(dim=1)
        return self.cls(fused)