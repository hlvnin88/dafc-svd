"""Tiện ích chung."""
import os, json, pickle, random, re
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import torch
from config import RESULTS_DIR, TABLES_DIR, MODELS_DIR

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def ensure_dir(p):
    p = Path(p); p.mkdir(parents=True, exist_ok=True); return p

def timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def save_csv(df, name, dir_path=None):
    d = ensure_dir(dir_path or TABLES_DIR)
    p = d / f'{name}.csv'; df.to_csv(p, index=False); return str(p)

def save_json(obj, name, dir_path=None):
    d = ensure_dir(dir_path or RESULTS_DIR)
    p = d / f'{name}.json'
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    return str(p)

def save_pickle(obj, name, dir_path=None):
    d = ensure_dir(dir_path or RESULTS_DIR)
    p = d / f'{name}.pkl'
    with open(p, 'wb') as f: pickle.dump(obj, f)
    return str(p)

def load_pickle(p):
    with open(p, 'rb') as f: return pickle.load(f)

def save_torch(model, name, dir_path=None):
    d = ensure_dir(dir_path or MODELS_DIR)
    p = d / f'{name}.pth'
    torch.save(model.state_dict(), p); return str(p)

def fmt_ms(mean, std, digits=3):
    if mean is None or std is None: return 'N/A'
    return f"{mean:.{digits}f} ± {std:.{digits}f}"

def print_section(title, char='=', length=80):
    print("\n" + char*length)
    print(f" {title} ".center(length, char))
    print(char*length)

def print_table(df, title, max_rows=30):
    if df is None or df.empty:
        print(f"⚠️  {title}: empty"); return
    print(f"\n📋 {title}")
    print(df.head(max_rows).to_string(index=False))
    if len(df) > max_rows:
        print(f"   ... ({len(df)} rows total)")

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def _mem_mb():
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / 1e6
    except Exception:
        return 0.0