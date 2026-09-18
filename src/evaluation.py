"""Bước 11: External test."""
import numpy as np
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from config import DEVICE
from src.dataloader import make_loader

@torch.no_grad()
def evaluate_model(model, X, y, bs=64):
    model.eval().to(DEVICE)
    loader = make_loader(X, y, bs, False)
    P, Y = [], []
    for xb, yb in loader:
        xb = [t.to(DEVICE) for t in xb]
        P.append(model(xb).cpu().numpy().flatten())
        Y.append(yb.numpy().flatten())
    P = np.concatenate(P); Y = np.concatenate(Y).astype(int)
    preds = (P >= 0.5).astype(int)
    return {'precision': precision_score(Y, preds, zero_division=0),
            'recall':    recall_score(Y, preds, zero_division=0),
            'f1':        f1_score(Y, preds, zero_division=0),
            'auc':       roc_auc_score(Y, P) if len(np.unique(Y))>1 else 0.5}