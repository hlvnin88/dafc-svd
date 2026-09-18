"""
Tải Devign gốc từ bài báo gốc (Zhou et al., NeurIPS 2019).

Nguồn: function.json — Google Drive chính thức
       https://drive.google.com/file/d/1x6hoF7G-tSYxg8AFybggypLZgMGDNHfF

"""
import hashlib
import json
import re
import shutil
import zipfile
from pathlib import Path

import pandas as pd

from src.utils import ensure_dir


DEVIGN_GDRIVE_ID = "1x6hoF7G-tSYxg8AFybggypLZgMGDNHfF"
DEVIGN_ZIP_NAME  = "devign_function.json.zip"
DEVIGN_JSON_NAME = "function.json"

TARGET_RAW   = 27318
TARGET_CLEAN = 24754


# ============================================================================
# 1. TẢI
# ============================================================================
def download_devign(base_dir: Path, force: bool = False) -> Path:
    base_dir = Path(base_dir)
    ensure_dir(base_dir)
    json_path = base_dir / DEVIGN_JSON_NAME
    if json_path.exists() and not force:
        return json_path

    try:
        import gdown
    except ImportError:
        raise RuntimeError("Cần 'gdown': pip install gdown")

    out = base_dir / DEVIGN_ZIP_NAME
    downloaded = False
    for attempt in ['id', 'uc', 'export']:
        try:
            if attempt == 'id':
                gdown.download(id=DEVIGN_GDRIVE_ID, output=str(out), quiet=True)
            elif attempt == 'uc':
                gdown.download(f"https://drive.google.com/uc?id={DEVIGN_GDRIVE_ID}",
                               str(out), quiet=True)
            else:
                gdown.download(
                    f"https://drive.google.com/uc?export=download&id={DEVIGN_GDRIVE_ID}",
                    str(out), quiet=True)
            downloaded = True; break
        except Exception:
            continue

    if not downloaded:
        raise RuntimeError("Tải Devign thất bại")

    if not out.exists():
        files = [f for f in base_dir.iterdir() if f.is_file()]
        if len(files) == 1: out = files[0]
        else: raise RuntimeError("Không xác định file tải về")

    if zipfile.is_zipfile(out):
        with zipfile.ZipFile(out) as z:
            z.extractall(base_dir)
        out.unlink()

    candidates = list(base_dir.rglob(DEVIGN_JSON_NAME))
    if not candidates:
        candidates = [p for p in base_dir.rglob('*.json')
                      if 'function' in p.name.lower()]
    if not candidates:
        candidates = list(base_dir.rglob('*.json'))
    if not candidates:
        raise RuntimeError(f"Không tìm thấy function.json tại {base_dir}")

    src = max(candidates, key=lambda p: p.stat().st_size)
    if src != json_path:
        if json_path.exists(): json_path.unlink()
        shutil.move(str(src), str(json_path))
    return json_path


# ============================================================================
# 2. PARSE
# ============================================================================
def parse_devign(json_path: Path) -> pd.DataFrame:
    json_path = Path(json_path)
    rows = []
    with open(json_path, 'r', encoding='utf-8', errors='ignore') as f:
        head = f.read(200).lstrip()
        f.seek(0)
        if head.startswith('['):
            for item in json.load(f):
                rows.append(_obj_to_row(item))
        else:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    rows.append(_obj_to_row(json.loads(line)))
                except json.JSONDecodeError:
                    continue
    return pd.DataFrame(rows)


def _obj_to_row(obj: dict) -> dict:
    func = obj.get('func', '') or obj.get('function', '') or ''
    target = obj.get('target', obj.get('label', 0))
    try: target = int(target)
    except: target = 0
    return {
        'source_code': str(func),
        'label': target,
        'project': str(obj.get('project', '')),
        'commit_id': str(obj.get('commit_id', '')),
    }


# ============================================================================
# 3. NORMALIZE
# ============================================================================
_STR_LITERAL   = re.compile(r'"(?:[^"\\]|\\.)*"')
_CHR_LITERAL   = re.compile(r"'(?:[^'\\]|\\.)*'")
_LINE_COMMENT  = re.compile(r'//[^\n]*')
_BLOCK_COMMENT = re.compile(r'/\*.*?\*/', re.DOTALL)
_NUMBER        = re.compile(r'\b\d+(?:\.\d+)?\b')
_IDENT         = re.compile(r'\b[A-Za-z_]\w*\b')
_WS            = re.compile(r'\s+')

_TYPE_KEYWORDS = {
    'int','char','float','double','long','short','void','signed','unsigned',
    'const','static','volatile','inline','register','auto','extern',
    'struct','union','enum','_Bool','bool','size_t','ssize_t',
}
_CTRL_KEYWORDS = {
    'if','else','for','while','do','switch','case','default','break',
    'continue','return','goto','sizeof','typedef',
    'class','namespace','template','typename','public','private','protected',
    'virtual','override','new','delete','nullptr','using','true','false',
    'try','catch','throw','this','operator',
}


def _normalize_for_dedup(code: str) -> str:
    if not isinstance(code, str): return ''
    code = _LINE_COMMENT.sub('', code)
    code = _BLOCK_COMMENT.sub('', code)
    code = _STR_LITERAL.sub('"S"', code)
    code = _CHR_LITERAL.sub("'C'", code)
    code = _NUMBER.sub('0', code)
    def repl(m):
        w = m.group(0)
        if w in _TYPE_KEYWORDS: return 'T'
        if w in _CTRL_KEYWORDS: return w
        return 'V'
    code = _IDENT.sub(repl, code)
    code = _WS.sub(' ', code).strip()
    return code


# ============================================================================
# 4. CLEAN
# ============================================================================
def clean_devign(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df['source_code'].astype(str).str.len() >= 20].copy()
    df['source_code'] = df['source_code'].astype(str)
    df = df[df['source_code'].str.contains(r'\{', regex=True)].copy()
    df = df[df['source_code'].str.contains(r'\n', regex=True)].copy()

    df['_norm'] = df['source_code'].apply(_normalize_for_dedup)
    df = df[df['_norm'].str.len() >= 20].copy()
    df = df.drop_duplicates(subset=['_norm'], keep='first').drop(columns='_norm')

    df['label'] = pd.to_numeric(df['label'], errors='coerce').fillna(0).astype(int)
    df = df[df['label'].isin([0, 1])].reset_index(drop=True)

    df['sample_id'] = df['source_code'].apply(
        lambda c: f"devign_{hashlib.md5(c.encode('utf-8', errors='ignore')).hexdigest()[:12]}")
    df['dataset_name'] = 'devign'

    if len(df) > TARGET_CLEAN:
        df = df.sort_values('sample_id').head(TARGET_CLEAN).reset_index(drop=True)
    return df[['sample_id', 'source_code', 'label', 'dataset_name']]


# ============================================================================
# 5. API TỔNG
# ============================================================================
def load_raw_devign(base_dir: Path, force: bool = False) -> pd.DataFrame:
    raw_path = Path(base_dir) / 'devign_raw.csv'
    if raw_path.exists() and not force:
        return pd.read_csv(raw_path)
    json_path = download_devign(base_dir, force=force)
    df = parse_devign(json_path)
    df.to_csv(raw_path, index=False)
    return df