"""Bước 2: Tiền xử lý."""
import re
import pandas as pd
from config import DATA_DIR, DATASET_JULIET
from src.utils import ensure_dir

_WS = re.compile(r'[ \t]+')
_NL = re.compile(r'\n{3,}')


def normalize_code(code):
    if not isinstance(code, str): return ''
    code = code.encode('utf-8', errors='ignore').decode('utf-8', errors='ignore')
    code = code.replace('\r\n','\n').replace('\r','\n').replace('\t','  ')
    code = _WS.sub(' ', code); code = _NL.sub('\n\n', code)
    return code.strip()


def _valid(code, label):
    if not isinstance(code, str) or len(code.strip()) < 20: return False
    try: int(label)
    except: return False
    return True


def run_preprocess(name, df):
    log = {'input': len(df)}
    df = df.dropna(subset=['source_code','label']).copy()
    log['after_dropna'] = len(df)
    df['source_code'] = df['source_code'].apply(normalize_code)
    mask = df.apply(lambda r: _valid(r['source_code'], r['label']), axis=1)
    df = df[mask].reset_index(drop=True)
    log['after_validity'] = len(df)

    # Juliet: dedup trên (code, label) — vì đã có cả good và bad
    # Devign/Big-Vul: dedup trên code
    subset = ['source_code', 'label'] if name == DATASET_JULIET else ['source_code']
    df = df.drop_duplicates(subset=subset, keep='first').reset_index(drop=True)
    log['after_dedup'] = len(df)

    df['label'] = df['label'].astype(int)
    df.loc[~df['label'].isin([0,1]), 'label'] = 0
    df = df[['sample_id','source_code','label','dataset_name']]
    out = ensure_dir(DATA_DIR / name / 'clean') / f'{name}_clean.csv'
    df.to_csv(out, index=False)
    print(f"✅ Clean {name}: {log}")
    return df