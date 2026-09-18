"""F7: Joern CPG (all edges) + GCN → 256."""
import numpy as np
from config import DATA_DIR
from src.utils import ensure_dir
from src.features.f3_ast import extract_f3

def extract_f7(df_full, name):
    Z = extract_f3(df_full, name, edge_type='ALL')
    out = ensure_dir(DATA_DIR / name / 'features' / 'F7')
    np.save(out / f'{name}_full_f7.npy', Z)
    return Z