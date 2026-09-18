"""F2: Program Slice + BiLSTM → 256.
Lấy slice text từ Joern cache, token hoá, train BiLSTM phân loại, dùng hidden state 256 làm feature.
"""
import json, os, re
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from config import (DATA_DIR, FUSION_DIM, BILSTM_EMBED_DIM, BILSTM_HIDDEN,
                     BILSTM_VOCAB_MAX, BILSTM_MAX_LEN, BILSTM_LR, BILSTM_EPOCHS,
                     DEVICE, MODELS_DIR)
from src.utils import ensure_dir
from src.bilstm import BiLSTMEncoder

_TOKEN_RE = re.compile(r'[A-Za-z_]\w*|\d+|\S')

def _slice_text(row, cache_dir):
    sid = str(row['sample_id']); cj = os.path.join(cache_dir, f'{sid}.json')
    if os.path.exists(cj):
        with open(cj) as f: d = json.load(f)
        parts = []
        for s in (d.get('slices') or []):
            if isinstance(s, dict): parts.append(str(s)[:2000])
        if parts: return ' '.join(parts)
    return str(row['source_code'])

class _DS(Dataset):
    def __init__(self, X, y): self.X, self.y = X, y
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]

def _build_vocab(texts, max_vocab):
    from collections import Counter
    cnt = Counter()
    for t in texts: cnt.update(_TOKEN_RE.findall(t))
    most = cnt.most_common(max_vocab - 1)
    vocab = {'<pad>': 0}
    for tok, _ in most: vocab[tok] = len(vocab)
    return vocab

def _encode(texts, vocab, max_len):
    out = np.zeros((len(texts), max_len), dtype=np.int64)
    for i, t in enumerate(texts):
        ids = [vocab.get(w, 0) for w in _TOKEN_RE.findall(t)][:max_len]
        out[i, :len(ids)] = ids
    return out

def _lengths(X):
    return np.array([max(1, (row != 0).sum()) for row in X])

def extract_f2(df_train, df_val, df_test, name):
    cache = DATA_DIR / name / 'joern_raw'
    tr_texts = [_slice_text(r, str(cache)) for _, r in df_train.iterrows()]
    va_texts = [_slice_text(r, str(cache)) for _, r in df_val.iterrows()]
    te_texts = [_slice_text(r, str(cache)) for _, r in df_test.iterrows()]

    vocab = _build_vocab(tr_texts + va_texts + te_texts, BILSTM_VOCAB_MAX)
    Xtr = _encode(tr_texts, vocab, BILSTM_MAX_LEN)
    Xva = _encode(va_texts, vocab, BILSTM_MAX_LEN)
    Xte = _encode(te_texts, vocab, BILSTM_MAX_LEN)
    ytr = df_train['label'].values.astype(np.float32)
    yva = df_val['label'].values.astype(np.float32)

    model = BiLSTMEncoder(len(vocab), BILSTM_EMBED_DIM, BILSTM_HIDDEN, FUSION_DIM).to(DEVICE)
    cls = nn.Linear(FUSION_DIM, 1).to(DEVICE)
    opt = optim.Adam(list(model.parameters()) + list(cls.parameters()), lr=BILSTM_LR)
    loss_fn = nn.BCEWithLogitsLoss()
    dl = DataLoader(_DS(torch.tensor(Xtr), torch.tensor(ytr)),
                    batch_size=32, shuffle=True)

    model.train(); cls.train()
    for ep in range(BILSTM_EPOCHS):
        for xb, yb in dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            h = model(xb, torch.tensor(_lengths(xb.cpu().numpy()), dtype=torch.long))
            loss = loss_fn(cls(h).view(-1), yb)
            loss.backward(); opt.step()

    ckpt = ensure_dir(MODELS_DIR / 'encoders') / f'{name}_f2_bilstm.pth'
    torch.save(model.state_dict(), ckpt)

    # Sinh features
    @torch.no_grad()
    def _embed(X):
        model.eval()
        out = []
        for i in range(0, len(X), 64):
            xb = torch.tensor(X[i:i+64]).to(DEVICE)
            lens = torch.tensor(_lengths(X[i:i+64]), dtype=torch.long)
            out.append(model(xb, lens).cpu().numpy())
        return np.concatenate(out, 0).astype(np.float32)

    Z_tr, Z_va, Z_te = _embed(Xtr), _embed(Xva), _embed(Xte)
    out = ensure_dir(DATA_DIR / name / 'features' / 'F2')
    np.save(out / f'{name}_train_f2.npy', Z_tr)
    np.save(out / f'{name}_val_f2.npy',   Z_va)
    np.save(out / f'{name}_test_f2.npy',  Z_te)
    return Z_tr, Z_va, Z_te