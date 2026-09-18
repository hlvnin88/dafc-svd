"""F10: 32 static security metrics → 256."""
import re
import numpy as np
import torch
import torch.nn as nn
from config import F10_METRIC_NAMES, FUSION_DIM, DATA_DIR, DEVICE, MODELS_DIR
from src.utils import ensure_dir

_DANGEROUS = ['strcpy','strcat','sprintf','gets','malloc','free','memcpy']

def _metrics(code):
    v = np.zeros(len(F10_METRIC_NAMES), dtype=np.float32)
    if not isinstance(code, str): return v
    lines = code.split('\n')
    def cnt(p): return len(re.findall(p, code))
    vals = {
        'num_lines': len(lines),
        'num_functions': len(re.findall(r'\b\w+\s+\w+\s*\([^;]*\)\s*\{', code)),
        'num_params': cnt(r','),
        'cyclomatic_complexity': 1 + cnt(r'\b(if|for|while|case|&&|\|\|)\b'),
        'nesting_depth': max([l.count('{') for l in lines] + [0]),
        'num_branches': cnt(r'\b(if|else|switch|case)\b'),
        'num_loops': cnt(r'\b(for|while|do)\b'),
        'num_returns': cnt(r'\breturn\b'),
        'num_dangerous_apis': sum(cnt(rf'\b{d}\b') for d in _DANGEROUS),
        'num_strcpy': cnt(r'\bstrcpy\b'), 'num_strcat': cnt(r'\bstrcat\b'),
        'num_sprintf': cnt(r'\bsprintf\b'), 'num_gets': cnt(r'\bgets\b'),
        'num_malloc': cnt(r'\bmalloc\b'), 'num_free': cnt(r'\bfree\b'),
        'num_memcpy': cnt(r'\bmemcpy\b'),
        'has_pointer': int('*' in code), 'num_pointer_derefs': cnt(r'\*'),
        'has_array': int('[' in code), 'num_array_index': cnt(r'\['),
        'has_ampersand': int('&' in code),
        'num_bit_ops': cnt(r'(<<|>>|&|\||\^|~)'),
        'has_union': int(bool(re.search(r'\bunion\b', code))),
        'has_void_ptr': int('void' in code and '*' in code),
        'has_bounds_check': int(bool(re.search(r'(if|while)\s*\([^)]*<[^)]*\)', code))),
        'has_null_check': int('NULL' in code or 'nullptr' in code),
        'has_size_check': int(bool(re.search(r'\bsizeof\b', code))),
        'num_arithmetic_ops': cnt(r'[\+\-\*/%]'),
        'num_assignments': cnt(r'='),
        'num_comparisons': cnt(r'(==|!=|<=|>=|<|>)'),
        'num_casts': cnt(r'\(\s*(int|char|float|double|void)\s*\*?\s*\)'),
        'num_goto': cnt(r'\bgoto\b'),
    }
    for i, n in enumerate(F10_METRIC_NAMES): v[i] = vals.get(n, 0)
    return v

def _train_projection(X_tr, y_tr, epochs=5):
    proj = nn.Sequential(nn.Linear(X_tr.shape[1], FUSION_DIM), nn.ReLU()).to(DEVICE)
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
            h = proj(xb); loss = loss_fn(cls(h).view(-1), yb)
            loss.backward(); opt.step()
    return proj

@torch.no_grad()
def _apply(proj, X):
    proj.eval(); out = []
    X_t = torch.tensor(X, dtype=torch.float32)
    for i in range(0, len(X), 64):
        out.append(proj(X_t[i:i+64].to(DEVICE)).cpu().numpy())
    return np.concatenate(out, 0).astype(np.float32)

def extract_f10(tr_codes, va_codes, te_codes, name, y_tr):
    X_tr = np.stack([_metrics(c) for c in tr_codes])
    X_va = np.stack([_metrics(c) for c in va_codes])
    X_te = np.stack([_metrics(c) for c in te_codes])
    proj = _train_projection(X_tr, y_tr)
    Z_tr = _apply(proj, X_tr); Z_va = _apply(proj, X_va); Z_te = _apply(proj, X_te)
    out = ensure_dir(DATA_DIR / name / 'features' / 'F10')
    np.save(out / f'{name}_train_f10.npy', Z_tr)
    np.save(out / f'{name}_val_f10.npy',   Z_va)
    np.save(out / f'{name}_test_f10.npy',  Z_te)
    torch.save(proj.state_dict(),
               ensure_dir(MODELS_DIR / 'encoders') / f'{name}_f10_proj.pth')
    return Z_tr, Z_va, Z_te