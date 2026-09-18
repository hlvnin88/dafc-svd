"""Bước 9–11: Train MMAF với 5 seeds."""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from config import (MMAF_LR, MMAF_WEIGHT_DECAY, MMAF_BATCH_SIZE, MMAF_EPOCHS,
                     MMAF_EARLY_STOP, RANDOM_SEEDS, DEVICE, MODELS_DIR)
from src.dataloader import make_loader
from src.mmf import MMAF
from src.utils import ensure_dir

def _train_one(in_dims, Xtr, ytr, Xva, yva, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    tr_loader = make_loader(Xtr, ytr, MMAF_BATCH_SIZE, True)
    va_loader = make_loader(Xva, yva, MMAF_BATCH_SIZE, False)
    model = MMAF(in_dims).to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr=MMAF_LR, weight_decay=MMAF_WEIGHT_DECAY)
    loss_fn = nn.BCELoss()

    best = float('inf'); best_state = None; wait = 0
    for _ in range(MMAF_EPOCHS):
        model.train()
        for xb, yb in tr_loader:
            xb = [t.to(DEVICE) for t in xb]; yb = yb.to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb); loss.backward(); opt.step()
        model.eval(); vl = 0.0; nb = 0
        with torch.no_grad():
            for xb, yb in va_loader:
                xb = [t.to(DEVICE) for t in xb]; yb = yb.to(DEVICE)
                vl += loss_fn(model(xb), yb).item(); nb += 1
        vl /= max(nb, 1)
        if vl < best:
            best = vl; wait = 0
            best_state = {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= MMAF_EARLY_STOP: break
    if best_state: model.load_state_dict(best_state)
    return model

@torch.no_grad()
def _predict(model, X, y, bs=64):
    model.eval().to(DEVICE)
    loader = make_loader(X, y, bs, False)
    P, Y = [], []
    for xb, yb in loader:
        xb = [t.to(DEVICE) for t in xb]
        P.append(model(xb).cpu().numpy().flatten())
        Y.append(yb.numpy().flatten())
    P = np.concatenate(P); Y = np.concatenate(Y).astype(int)
    return P, (P >= 0.5).astype(int), Y

def train_and_eval_combo(combo, Xtr, ytr, Xva, yva, Xte, yte,
                         run_seeds=RANDOM_SEEDS, ckpt_name=None):
    in_dims = [Xtr[0].shape[1]] * len(combo)
    per_val, per_te = [], []
    last_model = None
    for seed in run_seeds:
        model = _train_one(in_dims, Xtr, ytr, Xva, yva, seed)
        pv, prv, yv = _predict(model, Xva, yva)
        pt, prt, yt = _predict(model, Xte, yte)
        per_val.append({'precision': precision_score(yv, prv, zero_division=0),
                         'recall':    recall_score(yv, prv, zero_division=0),
                         'f1':        f1_score(yv, prv, zero_division=0),
                         'auc':       roc_auc_score(yv, pv) if len(np.unique(yv))>1 else 0.5})
        per_te.append({'precision': precision_score(yt, prt, zero_division=0),
                        'recall':    recall_score(yt, prt, zero_division=0),
                        'f1':        f1_score(yt, prt, zero_division=0),
                        'auc':       roc_auc_score(yt, pt) if len(np.unique(yt))>1 else 0.5})
        last_model = model
        if ckpt_name:
            torch.save(model.state_dict(),
                       ensure_dir(MODELS_DIR) / f'{ckpt_name}_seed{seed}.pth')
    keys = ['precision','recall','f1','auc']
    def _stats(lst):
        return ({k: float(np.mean([m[k] for m in lst])) for k in keys},
                {k: float(np.std([m[k] for m in lst]))  for k in keys})
    mv, sv = _stats(per_val); mt, st = _stats(per_te)
    return {'mean_val': mv, 'std_val': sv, 'per_seed_val': per_val,
            'mean_test': mt, 'std_test': st, 'per_seed_test': per_te,
            'last_model': last_model}