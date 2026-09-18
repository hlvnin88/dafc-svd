"""F8: CodeBERT 768 → 256. Linear projection được train cùng Devign train."""
import numpy as np
import torch
import torch.nn as nn
from config import CODEBERT_NAME, ENCODER_MAX_LEN, FUSION_DIM, DATA_DIR, DEVICE, MODELS_DIR
from src.utils import ensure_dir

_tok = None; _mdl = None

def _load():
    global _tok, _mdl
    if _tok is None:
        from transformers import AutoTokenizer, AutoModel
        print("⏳ Loading CodeBERT...")
        _tok = AutoTokenizer.from_pretrained(CODEBERT_NAME)
        _mdl = AutoModel.from_pretrained(CODEBERT_NAME).eval().to(DEVICE)
        print("✅ CodeBERT loaded")
    return _tok, _mdl

@torch.no_grad()
def _embed768(codes, batch=16):
    tok, mdl = _load()
    out = []
    for i in range(0, len(codes), batch):
        enc = tok(codes[i:i+batch], return_tensors='pt', truncation=True,
                  max_length=ENCODER_MAX_LEN, padding=True)
        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        h = mdl(**enc).last_hidden_state.mean(dim=1)
        out.append(h.cpu().numpy().astype(np.float32))
    return np.concatenate(out, 0)

def _train_projection(X_tr, y_tr, epochs=5):
    """Train Linear 768→256 như một encoder đơn giản."""
    proj = nn.Sequential(nn.Linear(768, FUSION_DIM), nn.ReLU()).to(DEVICE)
    cls = nn.Linear(FUSION_DIM, 1).to(DEVICE)
    opt = torch.optim.Adam(list(proj.parameters()) + list(cls.parameters()), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    X = torch.tensor(X_tr, dtype=torch.float32)
    y = torch.tensor(y_tr, dtype=torch.float32).view(-1)
    proj.train(); cls.train()
    for _ in range(epochs):
        for i in range(0, len(X), 64):
            xb = X[i:i+64].to(DEVICE); yb = y[i:i+64].to(DEVICE)
            opt.zero_grad()
            h = proj(xb)
            loss = loss_fn(cls(h).view(-1), yb)
            loss.backward(); opt.step()
    return proj

@torch.no_grad()
def _apply(proj, X):
    proj.eval()
    out = []
    X_t = torch.tensor(X, dtype=torch.float32)
    for i in range(0, len(X), 64):
        out.append(proj(X_t[i:i+64].to(DEVICE)).cpu().numpy())
    return np.concatenate(out, 0).astype(np.float32)

def extract_f8(tr_codes, va_codes, te_codes, name, y_tr):
    X_tr = _embed768(tr_codes)
    X_va = _embed768(va_codes)
    X_te = _embed768(te_codes)
    proj = _train_projection(X_tr, y_tr)
    Z_tr = _apply(proj, X_tr); Z_va = _apply(proj, X_va); Z_te = _apply(proj, X_te)
    out = ensure_dir(DATA_DIR / name / 'features' / 'F8')
    np.save(out / f'{name}_train_f8.npy', Z_tr)
    np.save(out / f'{name}_val_f8.npy',   Z_va)
    np.save(out / f'{name}_test_f8.npy',  Z_te)
    torch.save(proj.state_dict(),
               ensure_dir(MODELS_DIR / 'encoders') / f'{name}_f8_proj.pth')
    return Z_tr, Z_va, Z_te