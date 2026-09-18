"""Bước 6: Sinh 375 tổ hợp."""
import itertools
import pandas as pd
from config import (FEATURE_GROUPS, MIN_COMBO_SIZE, MAX_COMBO_SIZE, SELECTION_DIR)
from src.utils import ensure_dir

def generate_all_combos(groups=None):
    groups = groups or FEATURE_GROUPS
    out = []
    for r in range(MIN_COMBO_SIZE, MAX_COMBO_SIZE+1):
        out.extend(list(itertools.combinations(groups, r)))
    return out

def save_candidates(combos):
    rows = [{'combo_id': f'C{i:04d}', 'size': len(c), 'features': ' + '.join(c)}
            for i, c in enumerate(combos, 1)]
    df = pd.DataFrame(rows)
    df.to_csv(ensure_dir(SELECTION_DIR) / 'candidates_2to4.csv', index=False)
    print(f"💾 candidates_2to4.csv ({len(df)})")
    return df