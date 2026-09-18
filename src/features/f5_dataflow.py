"""F5: Joern data-flow + GCN → 256."""
import numpy as np
from config import DATA_DIR
from src.utils import ensure_dir
from src.features.f3_ast import extract_f3

def extract_f5(df_full, name):
    Z = extract_f3(df_full, name, edge_type='DDG')
    out = ensure_dir(DATA_DIR / name / 'features' / 'F5')
    np.save(out / f'{name}_full_f5.npy', Z)
    return Z