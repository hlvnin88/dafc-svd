#!/usr/bin/env python
"""Bước 1+2: Tải + preprocess + split IDs."""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd

from config import (DATASETS, print_config, TABLES_DIR,
                     DATASET_DEVIGN, DATASET_JULIET, DATASET_BIGVUL, DATA_DIR)
from src.data_loader import load_raw
from src.splits import create_or_load_splits
from src.utils import print_section, set_seed, save_csv, ensure_dir


def _clean_specific(name, df_raw):
    if name == DATASET_DEVIGN:
        from src.devign_orig import clean_devign
        return clean_devign(df_raw)

    elif name == DATASET_JULIET:
        # Juliet clean lấy từ FULL (bad + good), không từ raw public
        from src.juliet_nist import clean_juliet_from_full
        base = ensure_dir(DATA_DIR / 'juliet' / 'nist')
        return clean_juliet_from_full(base)

    elif name == DATASET_BIGVUL:
        from src.bigvul_orig import clean_bigvul
        return clean_bigvul(df_raw)

    return df_raw


def main(force=False):
    print_config(); set_seed(42)
    print_section("BƯỚC 1+2 — Dataset + Split cố định", '=')

    rows = []
    for name in DATASETS:
        print(f"\n▶️  {name}")

        df_raw = load_raw(name, force=force)
        n_raw = len(df_raw)

        df_clean = _clean_specific(name, df_raw)
        n_clean = len(df_clean)

        out = ensure_dir(DATA_DIR / name / 'clean') / f'{name}_clean.csv'
        df_clean.to_csv(out, index=False)

        tr, va, te = create_or_load_splits(name, df_clean, force=force)

        print(f"   Raw samples   : {n_raw:>8,}")
        print(f"   Clean samples : {n_clean:>8,}")
        print(f"   Split         : train={len(tr):,}, val={len(va):,}, test={len(te):,}")

        rows.append({
            'Dataset': name,
            'Raw samples': n_raw,
            'Clean samples': n_clean,
            'Train': len(tr), 'Val': len(va), 'Test': len(te),
            'Role': 'Full pipeline' if name == 'devign'
                    else ('Retrain Top-K' if name == 'juliet'
                          else 'External test'),
        })

    print()
    summary = pd.DataFrame(rows)
    save_csv(summary, 'dataset_stats', TABLES_DIR)
    print(summary.to_string(index=False))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--force', action='store_true')
    main(force=ap.parse_args().force)