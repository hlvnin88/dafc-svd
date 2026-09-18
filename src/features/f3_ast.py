"""F3: Joern AST + GCN → 256. Train GCN encoder trên Devign train."""
import json, os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from config import DATA_DIR, FUSION_DIM, GCN_HIDDEN, DEVICE, MODELS_DIR
from src.utils import ensure_dir
from src.gcn import GCNEncoder

_LABEL_VOCAB = {}

def _label_id(lbl):
    lbl = str(lbl)
    if lbl not in _LABEL_VOCAB: _LABEL_VOCAB[lbl] = len(_LABEL_VOCAB)
    return _LABEL_VOCAB[lbl]

def _build_from_cache(sid, cache_dir, edge_type):
    p = os.path.join(cache_dir, f'{sid}.json')
    if not os.path.exists(p): return None
    with open(p) as f: d = json.load(f)
    if d.get('status') != 'ok' or d.get('nodes') is None: return None
    nodes = pd.DataFrame(d['nodes']); rels = pd.DataFrame(d['rels'])
    if nodes.empty: return None
    labels = nodes['label'].tolist() if 'label' in nodes else ['?']*len(nodes)
    feat_ids = np.array([_label_id(l) for l in labels])
    if edge_type == 'ALL': sub = rels
    elif 'type' in rels.columns: sub = rels[rels['type'] == edge_type]
    else: sub = rels
    if {'src','dst'}.issubset(sub.columns) and not sub.empty:
        edges = np.stack([sub['src'].values, sub['dst'].values]).astype(np.int64)
    else:
        edges = np.zeros((2,0), dtype=np.int64)
    return feat_ids, edges, len(nodes)

class _GraphDS(Dataset):
    def __init__(self, graphs, labels):
        self.g = graphs; self.y = labels
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.g[i], self.y[i]

def _train_gcn_encoder(graphs, labels, vocab_n, epochs=8):
    model = GCNEncoder(vocab_n, GCN_HIDDEN, FUSION_DIM).to(DEVICE)
    cls = nn.Linear(FUSION_DIM, 1).to(DEVICE)
    opt = optim.Adam(list(model.parameters()) + list(cls.parameters()), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    model.train(); cls.train()
    for ep in range(epochs):
        for (feat_ids, edges), y in zip(graphs, labels):
            if feat_ids is None: continue
            x = torch.zeros(len(feat_ids), vocab_n)
            x[torch.arange(len(feat_ids)), torch.tensor(feat_ids)] = 1
            x = x.to(DEVICE)
            ei = torch.tensor(edges, dtype=torch.long).to(DEVICE)
            opt.zero_grad()
            h = model(x, ei).view(-1)
            loss = loss_fn(h, torch.tensor([float(y)], device=DEVICE))
            loss.backward(); opt.step()
    return model

def extract_f3(df_full, name, edge_type='AST'):
    cache = DATA_DIR / name / 'joern_raw'
    sids = df_full['sample_id'].astype(str).tolist()
    labels = df_full['label'].values.astype(np.float32)
    graphs = [_build_from_cache(sid, str(cache), edge_type) for sid in sids]
    for g in graphs:
        if g is not None:
            for fid in g[0]: _label_id(fid)
    vocab_n = max(len(_LABEL_VOCAB), 1)

    # Train encoder trên các graph có sẵn
    model = _train_gcn_encoder(graphs, labels, vocab_n)

    @torch.no_grad()
    def _embed(g):
        if g is None: return np.zeros(FUSION_DIM, dtype=np.float32)
        feat_ids, edges, n = g
        x = torch.zeros(n, vocab_n)
        x[torch.arange(n), torch.tensor(feat_ids)] = 1
        ei = torch.tensor(edges, dtype=torch.long).to(DEVICE)
        z = model(x.to(DEVICE), ei).squeeze(0).cpu().numpy()
        return z.astype(np.float32)

    Z = np.stack([_embed(g) for g in graphs])
    out = ensure_dir(DATA_DIR / name / 'features' / 'F3')
    np.save(out / f'{name}_full_f3.npy', Z)
    torch.save(model.state_dict(),
               ensure_dir(MODELS_DIR / 'encoders') / f'{name}_f3_gcn.pth')
    return Z