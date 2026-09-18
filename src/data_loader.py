"""Bước 1: Tải dataset gốc với schema chuẩn."""
import hashlib
from pathlib import Path
import pandas as pd

from config import DATA_DIR, DATASET_DEVIGN, DATASET_JULIET, DATASET_BIGVUL
from src.utils import ensure_dir

REQUIRED_COLS = ['sample_id', 'source_code', 'label', 'dataset_name']

TARGET_RAW = {
    DATASET_DEVIGN: 27318,
    DATASET_JULIET: 64099,
    DATASET_BIGVUL: 188636,
}


def _is_valid_cache(path, expected_rows=None) -> bool:
    if not path.exists():
        return False
    try:
        df = pd.read_csv(path)
        if not all(c in df.columns for c in REQUIRED_COLS):
            return False
        if len(df) == 0:
            return False
        if expected_rows is not None and len(df) != expected_rows:
            return False
        return True
    except Exception:
        return False


def _load_devign() -> pd.DataFrame:
    from src.devign_orig import load_raw_devign
    base = ensure_dir(DATA_DIR / 'devign' / 'orig')
    df = load_raw_devign(base, force=False)
    return df[['source_code', 'label']].copy()


def _load_juliet() -> pd.DataFrame:
    from src.juliet_nist import load_raw_juliet
    base = ensure_dir(DATA_DIR / 'juliet' / 'nist')
    df = load_raw_juliet(base, method='git', force=False)
    return df[['source_code', 'label']].copy()


def _load_bigvul() -> pd.DataFrame:
    from src.bigvul_orig import load_raw_bigvul
    base = ensure_dir(DATA_DIR / 'bigvul' / 'orig')
    df = load_raw_bigvul(base, force=False)
    return df[['source_code', 'label']].copy()


def _load_hf(name: str) -> pd.DataFrame:
    if name == DATASET_DEVIGN:
        return _load_devign()
    elif name == DATASET_JULIET:
        return _load_juliet()
    elif name == DATASET_BIGVUL:
        return _load_bigvul()
    else:
        raise ValueError(f"Unknown dataset: {name}")


def load_raw(name: str, force: bool = False) -> pd.DataFrame:
    expected = TARGET_RAW.get(name)
    raw_dir = ensure_dir(DATA_DIR / name / 'raw')
    path = raw_dir / f'{name}_raw.csv'

    if not force and _is_valid_cache(path, expected_rows=expected):
        return pd.read_csv(path)

    if path.exists():
        path.unlink()

    df = _load_hf(name)
    if 'sample_id' not in df.columns:
        df.insert(0, 'sample_id', [
            f"{name}_{hashlib.md5(str(c).encode('utf-8')).hexdigest()[:12]}"
            for c in df['source_code']])
    if 'dataset_name' not in df.columns:
        df['dataset_name'] = name
    df = df[REQUIRED_COLS]
    df.to_csv(path, index=False)
    return df