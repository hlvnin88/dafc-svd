"""Bước 2: Cố định split IDs."""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from config import (
    DATA_DIR, DATASET_BIGVUL, SPLIT_SEED,
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO, BIGVUL_TEST_RATIO,
)
from src.utils import ensure_dir


def _dir(name): return ensure_dir(DATA_DIR / name / 'splits')


def _save(name, ids_dict):
    d = _dir(name)
    for split, ids in ids_dict.items():
        p = d / f'{name}_{split}_ids.txt'
        with open(p, 'w') as f:
            for sid in ids: f.write(f'{sid}\n')


def _load(name):
    d = _dir(name); out = {}
    for split in ['train', 'val', 'test']:
        p = d / f'{name}_{split}_ids.txt'
        if not p.exists(): return None
        with open(p) as f: out[split] = [l.strip() for l in f if l.strip()]
    return out


def create_or_load_splits(name, df, force=False):
    ids = None if force else _load(name)
    if ids is None:
        if name == DATASET_BIGVUL:
            n_test = int(round(len(df) * BIGVUL_TEST_RATIO))
            _, test_df = train_test_split(
                df, test_size=n_test,
                stratify=df['label'], random_state=SPLIT_SEED)
            train_df = df.iloc[0:0]
            val_df = df.iloc[0:0]
        else:
            train_df, temp = train_test_split(
                df, test_size=(VAL_RATIO + TEST_RATIO),
                stratify=df['label'], random_state=SPLIT_SEED)
            val_df, test_df = train_test_split(
                temp, test_size=TEST_RATIO / (VAL_RATIO + TEST_RATIO),
                stratify=temp['label'], random_state=SPLIT_SEED)

        _save(name, {
            'train': train_df['sample_id'].tolist(),
            'val': val_df['sample_id'].tolist(),
            'test': test_df['sample_id'].tolist(),
        })
    else:
        idx = df.set_index('sample_id')
        train_df = idx.loc[[i for i in ids['train'] if i in idx.index]].reset_index()
        val_df = idx.loc[[i for i in ids['val'] if i in idx.index]].reset_index()
        test_df = idx.loc[[i for i in ids['test'] if i in idx.index]].reset_index()
    return train_df, val_df, test_df