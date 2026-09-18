"""F6: Joern call graph + GCN → 256."""
import numpy as np
from config import DATA_DIR
from src.utils import ensure_dir
from src.features.f3_ast import extract_f3

def extract_f6(df_full, name):
    Z = extract_f3(df_full, name, edge_type='CALL')
    out = ensure_dir(DATA_DIR / name / 'features' / 'F6')
    np.save(out / f'{name}_full_f6.npy', Z)
    return Z