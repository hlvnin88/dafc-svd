"""Bước 13–14: Random/Top-individual + external baselines"""
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
from config import (FEATURE_GROUPS, RANDOM_SELECTION_SEED, DEVICE, MODELS_DIR)
from src.utils import ensure_dir


def random_selection(groups=None, k=4, seed=RANDOM_SELECTION_SEED):
    groups = groups or FEATURE_GROUPS
    random.seed(seed)
    return tuple(sorted(random.sample(groups, k)))


def top_individual_selection(perf_dict, k=4):
    ranked = sorted(perf_dict.items(), key=lambda kv: kv[1], reverse=True)
    return tuple(sorted([g for g, _ in ranked[:k]]))


# ============================================================================
# CodeBERT fine-tune baseline
# ============================================================================
class CodeBERTBaseline(nn.Module):
    def __init__(self, name='microsoft/codebert-base', out_dim=1, dropout=0.3):
        super().__init__()
        from transformers import AutoModel
        self.bert = AutoModel.from_pretrained(name)
        self.cls = nn.Sequential(nn.Linear(768, 128), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(128, out_dim),
                                  nn.Sigmoid())
    def forward(self, input_ids, attention_mask):
        h = self.bert(input_ids=input_ids, attention_mask=attention_mask
                      ).last_hidden_state.mean(dim=1)
        return self.cls(h)


# ============================================================================
# DWF: CodeBERT + GraphCodeBERT + dynamic gate
# ============================================================================
class DWF(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()
        from transformers import AutoModel
        self.bert1 = AutoModel.from_pretrained('microsoft/codebert-base')
        self.bert2 = AutoModel.from_pretrained('microsoft/graphcodebert-base')
        self.gate = nn.Sequential(nn.Linear(768*2, 128), nn.ReLU(),
                                   nn.Linear(128, 2), nn.Softmax(dim=-1))
        self.cls = nn.Sequential(nn.Linear(768, 128), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(128, 1),
                                  nn.Sigmoid())
    def forward(self, i1, m1, i2, m2):
        h1 = self.bert1(input_ids=i1, attention_mask=m1).last_hidden_state.mean(1)
        h2 = self.bert2(input_ids=i2, attention_mask=m2).last_hidden_state.mean(1)
        g = self.gate(torch.cat([h1, h2], dim=-1))
        return self.cls(g[:, 0:1]*h1 + g[:, 1:2]*h2)


# ============================================================================
# CPG + RGCN
# ============================================================================
class RGCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_rels):
        super().__init__()
        self.self_lin = nn.Linear(in_dim, out_dim)
        self.rel_lins = nn.ModuleList([nn.Linear(in_dim, out_dim)
                                        for _ in range(num_rels)])
    def forward(self, x, edge_index, edge_type, num_rels):
        out = self.self_lin(x)
        for r in range(num_rels):
            mask = (edge_type == r)
            if mask.any():
                src = edge_index[0][mask]; dst = edge_index[1][mask]
                agg = torch.zeros_like(x).index_add_(0, dst, self.rel_lins[r](x[src]))
                out = out + agg
        return torch.relu(out)


class CPG_RGCN(nn.Module):
    def __init__(self, node_in_dim, hidden=128, num_rels=1, out_dim=1, dropout=0.3):
        super().__init__()
        self.rgcn1 = RGCNLayer(node_in_dim, hidden, num_rels)
        self.rgcn2 = RGCNLayer(hidden, hidden, num_rels)
        self.cls = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(64, out_dim),
                                  nn.Sigmoid())
    def forward(self, x, edge_index, edge_type, num_rels, batch_vec):
        h = self.rgcn1(x, edge_index, edge_type, num_rels)
        h = self.rgcn2(h, edge_index, edge_type, num_rels)
        B = int(batch_vec.max().item()) + 1
        pooled = torch.zeros(B, h.size(1), device=h.device)
        cnt = torch.zeros(B, 1, device=h.device)
        pooled.index_add_(0, batch_vec, h)
        cnt.index_add_(0, batch_vec, torch.ones_like(h[:, :1]))
        pooled = pooled / cnt.clamp(min=1)
        return self.cls(pooled)


def train_text_baseline(model, train_loader, val_loader,
                        kind='codebert', lr=2e-5, epochs=3, patience=2):
    model.to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = nn.BCELoss()
    best = float('inf'); best_state = None; wait = 0
    for ep in range(epochs):
        model.train()
        for batch in train_loader:
            if kind == 'codebert':
                i1, m1, y = batch
                out = model(i1.to(DEVICE), m1.to(DEVICE))
                y = y.to(DEVICE)
            else:  # dwf
                i1, m1, i2, m2, y = batch
                out = model(i1.to(DEVICE), m1.to(DEVICE),
                            i2.to(DEVICE), m2.to(DEVICE))
                y = y.to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(out.view(-1), y.view(-1))
            loss.backward(); opt.step()
        model.eval(); vl = 0.0; nb = 0
        with torch.no_grad():
            for batch in val_loader:
                if kind == 'codebert':
                    i1, m1, y = batch
                    out = model(i1.to(DEVICE), m1.to(DEVICE))
                else:
                    i1, m1, i2, m2, y = batch
                    out = model(i1.to(DEVICE), m1.to(DEVICE),
                                i2.to(DEVICE), m2.to(DEVICE))
                vl += loss_fn(out.view(-1), y.to(DEVICE).view(-1)).item(); nb += 1
        vl /= max(nb, 1)
        if vl < best:
            best = vl; wait = 0
            best_state = {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience: break
    if best_state: model.load_state_dict(best_state)
    return model


@torch.no_grad()
def eval_text_baseline(model, loader, kind='codebert'):
    model.eval().to(DEVICE); P, Y = [], []
    for batch in loader:
        if kind == 'codebert':
            i1, m1, y = batch
            out = model(i1.to(DEVICE), m1.to(DEVICE))
        else:
            i1, m1, i2, m2, y = batch
            out = model(i1.to(DEVICE), m1.to(DEVICE),
                        i2.to(DEVICE), m2.to(DEVICE))
        P.append(out.view(-1).cpu().numpy()); Y.append(y.numpy().flatten())
    P = np.concatenate(P); Y = np.concatenate(Y).astype(int)
    preds = (P >= 0.5).astype(int)
    return {'precision': precision_score(Y, preds, zero_division=0),
            'recall':    recall_score(Y, preds, zero_division=0),
            'f1':        f1_score(Y, preds, zero_division=0),
            'auc':       roc_auc_score(Y, P) if len(np.unique(Y))>1 else 0.5}


# ============================================================================
# CPG + RGCN train/eval (dùng F7 như nút 1-graph)
# ============================================================================
def _to_graph(F, y):
    n, d = F.shape
    x = torch.tensor(F, dtype=torch.float32)
    ei = torch.zeros((2, 0), dtype=torch.long)
    et = torch.zeros(0, dtype=torch.long)
    bv = torch.arange(n, dtype=torch.long)
    yt = torch.tensor(y, dtype=torch.float32)
    return x, ei, et, bv, yt


def train_cpg_rgcn(node_dim, F_tr, y_tr, F_va, y_va, epochs=10, patience=3):
    model = CPG_RGCN(node_dim, hidden=128, num_rels=1).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    loss_fn = nn.BCELoss()
    x_tr, ei_tr, et_tr, bv_tr, yt_tr = _to_graph(F_tr, y_tr)
    x_va, ei_va, et_va, bv_va, yt_va = _to_graph(F_va, y_va)
    x_tr = x_tr.to(DEVICE); ei_tr = ei_tr.to(DEVICE); et_tr = et_tr.to(DEVICE)
    bv_tr = bv_tr.to(DEVICE); yt_tr = yt_tr.to(DEVICE)
    x_va = x_va.to(DEVICE); ei_va = ei_va.to(DEVICE); et_va = et_va.to(DEVICE)
    bv_va = bv_va.to(DEVICE); yt_va = yt_va.to(DEVICE)
    best = float('inf'); best_state = None; wait = 0
    for ep in range(epochs):
        model.train(); opt.zero_grad()
        out = model(x_tr, ei_tr, et_tr, 1, bv_tr).view(-1)
        loss = loss_fn(out, yt_tr); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            out_v = model(x_va, ei_va, et_va, 1, bv_va).view(-1)
            vl = loss_fn(out_v, yt_va).item()
        if vl < best:
            best = vl; wait = 0
            best_state = {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience: break
    if best_state: model.load_state_dict(best_state)
    return model


@torch.no_grad()
def eval_cpg_rgcn(model, F, y):
    model.eval().to(DEVICE)
    x, ei, et, bv, _ = _to_graph(F, y)
    out = model(x.to(DEVICE), ei.to(DEVICE), et.to(DEVICE), 1,
                bv.to(DEVICE)).view(-1).cpu().numpy()
    preds = (out >= 0.5).astype(int)
    return {'precision': precision_score(y, preds, zero_division=0),
            'recall':    recall_score(y, preds, zero_division=0),
            'f1':        f1_score(y, preds, zero_division=0),
            'auc':       roc_auc_score(y, out) if len(np.unique(y))>1 else 0.5}