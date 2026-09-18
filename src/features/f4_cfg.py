"""F4: Joern CFG + GCN → 256."""
import numpy as np
from config import DATA_DIR
from src.utils import ensure_dir
from src.features.f3_ast import extract_f3

def extract_f4(df_full, name):
    Z = extract_f3(df_full, name, edge_type='CFG')
    out = ensure_dir(DATA_DIR / name / 'features' / 'F4')
    np.save(out / f'{name}_full_f4.npy', Z)
    return Z