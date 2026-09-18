"""Bước 4: Huấn luyện F1–F10 độc lập."""
import time
import tracemalloc
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from config import (IND_HIDDEN, IND_DROPOUT, IND_LR, IND_WEIGHT_DECAY,
                     IND_BATCH_SIZE, IND_EPOCHS, IND_EARLY_STOP, IND_THRESHOLD,
                     RANDOM_SEEDS, DEVICE, MODELS_DIR, RESOURCE_LOGS)
from src.utils import ensure_dir


class IndClassifier(nn.Module):
    def __init__(self, in_dim=256, hidden=IND_HIDDEN, dropout=IND_DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1), nn.Sigmoid())
    def forward(self, x): return self.net(x)


def _train_one(Xtr, ytr, Xva, yva, Xte, yte, group, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    dev = DEVICE

    Xtr_t = torch.tensor(Xtr, dtype=torch.float32).to(dev)
    ytr_t = torch.tensor(ytr, dtype=torch.float32).view(-1, 1).to(dev)
    Xva_t = torch.tensor(Xva, dtype=torch.float32).to(dev)
    yva_t = torch.tensor(yva, dtype=torch.float32).view(-1, 1).to(dev)
    Xte_t = torch.tensor(Xte, dtype=torch.float32).to(dev)

    model = IndClassifier(Xtr.shape[1]).to(dev)
    opt = optim.AdamW(model.parameters(), lr=IND_LR, weight_decay=IND_WEIGHT_DECAY)
    loss_fn = nn.BCELoss()
    dl = DataLoader(TensorDataset(Xtr_t, ytr_t), batch_size=IND_BATCH_SIZE, shuffle=True)

    best = float('inf'); best_state = None; wait = 0
    for _ in range(IND_EPOCHS):
        model.train()
        for xb, yb in dl:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vl = loss_fn(model(Xva_t), yva_t).item()
        if vl < best:
            best = vl; wait = 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= IND_EARLY_STOP: break
    if best_state: model.load_state_dict(best_state)

    # === Đo inference time + peak memory ===
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    tracemalloc.start()
    model.eval()
    t0 = time.perf_counter()
    with torch.no_grad():
        _ = model(Xva_t)
        _ = model(Xte_t)
    elapsed = time.perf_counter() - t0
    n_meas = len(Xva) + len(Xte)
    ms_per_sample = elapsed / max(n_meas, 1) * 1000.0

    if torch.cuda.is_available():
        peak_bytes = torch.cuda.max_memory_allocated()
    else:
        _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mb = peak_bytes / (1024 ** 2)

    # Predictions
    with torch.no_grad():
        pv = model(Xva_t).cpu().numpy().flatten()
        pt = model(Xte_t).cpu().numpy().flatten()
    pvb = (pv >= IND_THRESHOLD).astype(int)
    ptb = (pt >= IND_THRESHOLD).astype(int)

    mv = {'precision': precision_score(yva, pvb, zero_division=0),
          'recall':    recall_score(yva, pvb, zero_division=0),
          'f1':        f1_score(yva, pvb, zero_division=0),
          'auc':       roc_auc_score(yva, pv) if len(np.unique(yva)) > 1 else 0.5}
    mt = {'precision': precision_score(yte, ptb, zero_division=0),
          'recall':    recall_score(yte, ptb, zero_division=0),
          'f1':        f1_score(yte, ptb, zero_division=0),
          'auc':       roc_auc_score(yte, pt) if len(np.unique(yte)) > 1 else 0.5}

    torch.save(model.state_dict(),
               ensure_dir(MODELS_DIR / 'individual') / f'{group}_seed{seed}.pth')

    return (mv, mt, pv, pvb, pt, ptb,
            {'inference_ms_per_sample': ms_per_sample,
             'peak_memory_mb': peak_mb,
             'n_samples_measured': n_meas})


def train_all_individuals(features_by_group, y_tr, y_va, y_te, dataset_name='devign'):
    import pandas as pd

    results = {}
    for g, d in features_by_group.items():
        n_tr = min(len(d['train']), len(y_tr))
        n_va = min(len(d['val']),   len(y_va))
        n_te = min(len(d['test']),  len(y_te))
        Xtr = d['train'][:n_tr]; ytr = y_tr[:n_tr]
        Xva = d['val'][:n_va];   yva = y_va[:n_va]
        Xte = d['test'][:n_te];  yte = y_te[:n_te]

        per_val, per_te = [], []
        val_p, val_pr, te_p, te_pr = {}, {}, {}, {}
        resources = []
        for seed in RANDOM_SEEDS:
            mv, mt, pv, pvb, pt, ptb, res = _train_one(
                Xtr, ytr, Xva, yva, Xte, yte, g, seed)
            per_val.append(mv); per_te.append(mt)
            val_p[seed] = pv; val_pr[seed] = pvb
            te_p[seed] = pt; te_pr[seed] = ptb
            resources.append(res)

        # Ghi resource log (cho Bảng #4 Cost components)
        avg_ms = float(np.mean([r['inference_ms_per_sample'] for r in resources]))
        avg_mem = float(np.mean([r['peak_memory_mb'] for r in resources]))
        row = pd.DataFrame([{
            'dataset': dataset_name, 'group': g,
            'extraction_time': 0.0,     # giữ nguyên giá trị cũ nếu đã có
            'inference_time': avg_ms,
            'dimension': 256,
            'memory_mb': avg_mem,
            'failure_rate': 0.0,
        }])
        p = RESOURCE_LOGS / f'{g}.csv'
        if p.exists():
            old = pd.read_csv(p)
            # Giữ extraction_time/failure_rate cũ nếu có
            if not old.empty and 'extraction_time' in old.columns:
                row['extraction_time'] = float(old['extraction_time'].iloc[-1])
            if not old.empty and 'failure_rate' in old.columns:
                row['failure_rate'] = float(old['failure_rate'].iloc[-1])
            row = pd.concat([old, row], ignore_index=True)
        row.to_csv(p, index=False)

        keys = ['precision','recall','f1','auc']
        mv_mean = {k: float(np.mean([m[k] for m in per_val])) for k in keys}
        mv_std  = {k: float(np.std([m[k] for m in per_val]))  for k in keys}
        mt_mean = {k: float(np.mean([m[k] for m in per_te]))  for k in keys}
        mt_std  = {k: float(np.std([m[k] for m in per_te]))   for k in keys}

        results[g] = {'per_seed_val': per_val, 'per_seed_test': per_te,
                      'mean_val': mv_mean, 'std_val': mv_std,
                      'mean_test': mt_mean, 'std_test': mt_std,
                      'val_probs_by_seed': val_p, 'val_preds_by_seed': val_pr,
                      'test_probs_by_seed': te_p, 'test_preds_by_seed': te_pr,
                      'resources': {'inference_ms_per_sample': avg_ms,
                                     'peak_memory_mb': avg_mem}}

        print(f"  {g}: F1(val)={mv_mean['f1']:.3f}±{mv_std['f1']:.3f}  "
              f"AUC={mv_mean['auc']:.3f}±{mv_std['auc']:.3f}  "
              f"inf={avg_ms:.3f}ms  mem={avg_mem:.0f}MB")
    return results