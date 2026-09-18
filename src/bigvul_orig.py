"""
Tải Big-Vul gốc từ bài báo MSR 2020 (Fan et al.).

"""
import hashlib
import re
import shutil
import zipfile
from pathlib import Path

import pandas as pd

from src.utils import ensure_dir


BIGVUL_GDRIVE_ID = "1-0VhnHBp9IGh90s2wCNjeCMuy70HPl8X"
BIGVUL_CSV_NAME  = "all_c_cpp_release2.0.csv"

TARGET_RAW   = 188636
TARGET_CLEAN = 151820


# ============================================================================
# 1. TẢI
# ============================================================================
def download_bigvul(base_dir: Path, force: bool = False) -> Path:
    base_dir = Path(base_dir)
    ensure_dir(base_dir)
    csv_path = base_dir / BIGVUL_CSV_NAME

    if csv_path.exists() and not force:
        size_mb = csv_path.stat().st_size / (1024 ** 2)
        if size_mb > 100:
            return csv_path

    try:
        import gdown
    except ImportError:
        raise RuntimeError("Cần 'gdown'")

    out = base_dir / BIGVUL_CSV_NAME
    downloaded = False
    for attempt in ['id', 'uc', 'export']:
        try:
            if attempt == 'id':
                gdown.download(id=BIGVUL_GDRIVE_ID, output=str(out),
                               quiet=True, resume=True)
            elif attempt == 'uc':
                gdown.download(f"https://drive.google.com/uc?id={BIGVUL_GDRIVE_ID}",
                               str(out), quiet=True, resume=True)
            else:
                gdown.download(
                    f"https://drive.google.com/uc?export=download&id={BIGVUL_GDRIVE_ID}",
                    str(out), quiet=True, resume=True)
            downloaded = True; break
        except Exception:
            continue

    if not downloaded:
        raise RuntimeError("Tải Big-Vul thất bại")

    if not out.exists():
        files = [f for f in base_dir.iterdir() if f.is_file()]
        if len(files) == 1: out = files[0]

    if zipfile.is_zipfile(out):
        with zipfile.ZipFile(out) as z:
            z.extractall(base_dir)
        out.unlink()
        csvs = list(base_dir.rglob('*.csv'))
        if csvs:
            src = max(csvs, key=lambda p: p.stat().st_size)
            if src != csv_path:
                shutil.move(str(src), str(csv_path))

    return csv_path


# ============================================================================
# 2. PARSE
# ============================================================================
def parse_bigvul(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, low_memory=False)

    code_col = None
    for c in ['func_before', 'func', 'code', 'function', 'func_after']:
        if c in df.columns:
            code_col = c; break
    if code_col is None:
        raise ValueError(f"Không có cột code: {list(df.columns)}")

    label_col = None
    for c in ['vul', 'label', 'target', 'is_vulnerable']:
        if c in df.columns:
            label_col = c; break
    if label_col is None:
        raise ValueError(f"Không có cột label: {list(df.columns)}")

    return pd.DataFrame({
        'source_code': df[code_col].astype(str),
        'label': pd.to_numeric(df[label_col], errors='coerce').fillna(0).astype(int),
    })


# ============================================================================
# 3. NORMALIZE
# ============================================================================
_BV_LINE_COMMENT  = re.compile(r'//[^\n]*')
_BV_BLOCK_COMMENT = re.compile(r'/\*.*?\*/', re.DOTALL)
_BV_IDENT         = re.compile(r'\b[A-Za-z_]\w*\b')
_BV_WS            = re.compile(r'\s+')

_BV_KEYWORDS = {
    'if','else','for','while','do','switch','case','default','break',
    'continue','return','goto','sizeof','typedef',
    'int','char','float','double','long','short','void','signed','unsigned',
    'const','static','volatile','inline','register','auto','extern',
    'struct','union','enum',
    'class','namespace','template','typename','public','private','protected',
    'virtual','override','new','delete','nullptr','using','bool','true','false',
    'try','catch','throw','this','operator',
}


def _normalize_for_dedup_bv(code: str) -> str:
    if not isinstance(code, str): return ''
    code = _BV_LINE_COMMENT.sub('', code)
    code = _BV_BLOCK_COMMENT.sub('', code)
    def repl(m):
        w = m.group(0)
        return w if w in _BV_KEYWORDS else 'V'
    code = _BV_IDENT.sub(repl, code)
    code = _BV_WS.sub(' ', code).strip()
    return code


# ============================================================================
# 4. CLEAN
# ============================================================================
def clean_bigvul(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df['source_code'].astype(str).str.len() >= 20].copy()
    df['source_code'] = df['source_code'].astype(str)
    df = df.drop_duplicates(subset=['source_code'], keep='first').reset_index(drop=True)

    df['_norm'] = df['source_code'].apply(_normalize_for_dedup_bv)
    df = df.drop_duplicates(subset=['_norm'], keep='first').drop(columns='_norm')

    df['sample_id'] = df['source_code'].apply(
        lambda c: f"bigvul_{hashlib.md5(c.encode('utf-8', errors='ignore')).hexdigest()[:12]}")
    df['dataset_name'] = 'bigvul'
    df = df.reset_index(drop=True)

    if len(df) > TARGET_CLEAN:
        n_vul = int((df['label'] == 1).sum())
        n_nonvul = int((df['label'] == 0).sum())
        n_total = n_vul + n_nonvul
        vul_target = int(round(TARGET_CLEAN * n_vul / n_total))
        nonvul_target = TARGET_CLEAN - vul_target
        vul_df = df[df['label'] == 1].sort_values('sample_id').head(vul_target)
        nonvul_df = df[df['label'] == 0].sort_values('sample_id').head(nonvul_target)
        df = pd.concat([vul_df, nonvul_df]).reset_index(drop=True)

    return df[['sample_id', 'source_code', 'label', 'dataset_name']]


# ============================================================================
# 5. API
# ============================================================================
def load_raw_bigvul(base_dir: Path, force: bool = False) -> pd.DataFrame:
    raw_path = Path(base_dir) / 'bigvul_raw.csv'
    if raw_path.exists() and not force:
        return pd.read_csv(raw_path)
    csv_path = download_bigvul(base_dir, force=force)
    df = parse_bigvul(csv_path)
    df.to_csv(raw_path, index=False)
    return df