"""
Tải và parse Juliet Test Suite 1.3 gốc (NIST).

"""
import hashlib
import os
import random
import re
import shutil
import subprocess
import time
import zipfile
from pathlib import Path

import pandas as pd

from src.utils import ensure_dir


# ============================================================================
# CẤU HÌNH
# ============================================================================
GITHUB_MIRRORS = [
    "https://github.com/arichardson/juliet-test-suite-c.git",
    "https://gitlab.com/samate/juliet-test-suite-1.3.git",
]
NIST_ZIP_URLS = [
    "https://samate.nist.gov/SARD/downloads/test-suites/"
    "2017-10-01-juliet-test-suite-for-c-cplusplus-v1-3.zip",
]

JULIET_REPO_DIR = 'juliet_repo'
SRC_EXT = {'.c', '.cpp'}

TARGET_RAW   = 64099
TARGET_CLEAN = 58420

MIN_ACCEPTABLE_FILES = int(TARGET_RAW * 0.85)


# ============================================================================
# TÌM GIT
# ============================================================================
GIT_PATH_CANDIDATES = [
    'git',
    r'C:\Program Files\Git\bin\git.exe',
    r'C:\Program Files (x86)\Git\bin\git.exe',
    r'C:\Program Files\Git\cmd\git.exe',
    r'C:\Users\PC\AppData\Local\Programs\Git\bin\git.exe',
]


def _find_git():
    import shutil as _sh
    for p in GIT_PATH_CANDIDATES:
        try:
            if _sh.which(p):
                return p
            if Path(p).exists():
                return p
        except Exception:
            continue
    return None


def _robust_rmtree(path: Path, max_retry: int = 3):
    path = Path(path)
    if not path.exists():
        return
    for _ in range(max_retry):
        try:
            if os.name == 'nt':
                subprocess.run(
                    ['cmd', '/c', 'rmdir', '/s', '/q', str(path)],
                    capture_output=True, timeout=180)
            else:
                shutil.rmtree(path, ignore_errors=True)
            if not path.exists():
                return
        except Exception:
            pass
        try:
            for p in path.rglob('*'):
                if p.is_file():
                    try:
                        os.chmod(p, 0o777)
                    except Exception:
                        pass
            shutil.rmtree(path, ignore_errors=True)
            if not path.exists():
                return
        except Exception:
            pass
        time.sleep(1.0)
    shutil.rmtree(path, ignore_errors=True)


# ============================================================================
# 1. TẢI
# ============================================================================
def download_juliet(base_dir: Path, method: str = 'git') -> Path:
    base_dir = Path(base_dir)
    repo_dir = base_dir / JULIET_REPO_DIR

    if (repo_dir / 'testcases').exists():
        n = _count_test_files(repo_dir)
        if n >= MIN_ACCEPTABLE_FILES:
            return repo_dir

    if repo_dir.exists():
        _robust_rmtree(repo_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    if method == 'zip':
        _try_zip_sources(base_dir, repo_dir)
        return repo_dir

    for mirror in GITHUB_MIRRORS:
        try:
            _clone_git(repo_dir, mirror_url=mirror)
            n = _count_test_files(repo_dir)
            if n >= MIN_ACCEPTABLE_FILES:
                return repo_dir
            _robust_rmtree(repo_dir)
        except Exception:
            _robust_rmtree(repo_dir)
            continue

    try:
        _try_zip_sources(base_dir, repo_dir)
        return repo_dir
    except Exception as e:
        raise RuntimeError(f"Không tải được Juliet: {e}")


def _clone_git(repo_dir: Path, mirror_url: str = None):
    if mirror_url is None:
        mirror_url = GITHUB_MIRRORS[0]
    git = _find_git()
    if git is None:
        raise RuntimeError("Không tìm thấy git.")
    config_args = [
        '-c', 'http.postBuffer=524288000',
        '-c', 'http.lowSpeedLimit=0',
        '-c', 'http.lowSpeedTime=999999',
        '-c', 'core.compression=0',
        '-c', 'http.version=HTTP/1.1',
    ]
    cmd = ([git] + config_args +
           ['clone', '--depth', '1', '--single-branch', '--no-tags',
            mirror_url, str(repo_dir)])
    try:
        result = subprocess.run(cmd, check=False, capture_output=True,
                                text=True, timeout=3600)
        if result.returncode != 0:
            raise RuntimeError("Git clone fail")
    except subprocess.TimeoutExpired:
        raise RuntimeError("Git clone timeout")


def _try_zip_sources(base_dir: Path, repo_dir: Path):
    for url in NIST_ZIP_URLS:
        try:
            _download_zip(url, base_dir, repo_dir)
            return
        except Exception:
            continue
    raise RuntimeError("Không tải được zip NIST")


def _download_zip(url: str, base_dir: Path, repo_dir: Path):
    import urllib.request
    zip_path = base_dir / 'juliet.zip'
    req = urllib.request.Request(url, headers={
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                      'AppleWebKit/537.36 (KHTML, like Gecko) '
                      'Chrome/120.0.0.0 Safari/537.36',
        'Accept': '*/*',
    })
    with urllib.request.urlopen(req, timeout=300) as resp, \
            open(zip_path, 'wb') as f:
        shutil.copyfileobj(resp, f)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(base_dir)
    zip_path.unlink()
    candidates = [d for d in base_dir.iterdir()
                  if d.is_dir() and (d / 'testcases').exists()]
    if not candidates:
        raise RuntimeError("Không có testcases/")
    if candidates[0] != repo_dir:
        if repo_dir.exists():
            _robust_rmtree(repo_dir)
        candidates[0].rename(repo_dir)


def _count_test_files(repo_dir: Path) -> int:
    tc = Path(repo_dir) / 'testcases'
    if not tc.exists():
        return 0
    n = 0
    for ext in SRC_EXT:
        n += sum(1 for _ in tc.rglob(f'*{ext}'))
    support = tc / 'testcasesupport'
    if support.exists():
        n -= sum(1 for _ in support.rglob('*.c'))
        n -= sum(1 for _ in support.rglob('*.cpp'))
    return n


# ============================================================================
# 2. PARSE MACRO BLOCKS
# ============================================================================
_MACRO_RE  = re.compile(r'^\s*#\s*ifndef\s+(OMITBAD|OMITGOOD)\b')
_ANY_IF_RE = re.compile(r'^\s*#\s*(if|ifdef|ifndef)\b')
_ENDIF_RE  = re.compile(r'^\s*#\s*endif\b')


def extract_macro_blocks(code: str) -> dict:
    lines = code.split('\n')
    result = {'bad': None, 'good': None}
    i = 0
    while i < len(lines):
        m = _MACRO_RE.match(lines[i])
        if not m:
            i += 1
            continue
        macro = m.group(1)
        depth = 1
        j = i + 1
        content = []
        while j < len(lines) and depth > 0:
            l = lines[j]
            if _ANY_IF_RE.match(l):
                depth += 1
            elif _ENDIF_RE.match(l):
                depth -= 1
                if depth == 0:
                    break
            if depth > 0:
                content.append(l)
            j += 1
        text = '\n'.join(content).strip()
        if macro == 'OMITBAD':
            result['bad'] = text
        else:
            result['good'] = text
        i = j + 1
    return result


# ============================================================================
# 3. PARSE FULL (bad + good)
# ============================================================================
def _hash(s: str) -> str:
    return hashlib.md5(s.encode('utf-8', errors='ignore')).hexdigest()[:12]


def parse_juliet_full(repo_dir: Path) -> pd.DataFrame:
    """
    Parse CẢ bad và good cho mọi file → DataFrame full (~191,450 samples).
    """
    repo_dir = Path(repo_dir)
    tc_dir = repo_dir / 'testcases'
    if not tc_dir.exists():
        raise FileNotFoundError(f"Không có {tc_dir}")

    all_files = []
    for ext in SRC_EXT:
        all_files.extend(tc_dir.rglob(f'*{ext}'))
    all_files = [f for f in all_files if 'testcasesupport' not in f.parts]
    src_files = sorted(all_files, key=lambda p: str(p))

    rows = []
    for path in src_files:
        try:
            code = path.read_text(encoding='utf-8', errors='ignore')
        except Exception:
            continue

        blocks = extract_macro_blocks(code)
        rel = str(path.relative_to(tc_dir))

        if blocks['bad'] and len(blocks['bad'].strip()) >= 20:
            rows.append({'sample_id': f"juliet_{_hash(blocks['bad'])}",
                         'source_code': blocks['bad'].strip(),
                         'label': 1, 'dataset_name': 'juliet',
                         'origin_file': rel, 'variant': 'bad'})
        if blocks['good'] and len(blocks['good'].strip()) >= 20:
            rows.append({'sample_id': f"juliet_{_hash(blocks['good'])}",
                         'source_code': blocks['good'].strip(),
                         'label': 0, 'dataset_name': 'juliet',
                         'origin_file': rel, 'variant': 'good'})

    return pd.DataFrame(rows)


# ============================================================================
# 4. RAW PUBLIC — 1 variant/file
# ============================================================================
def parse_juliet_raw(repo_dir: Path) -> pd.DataFrame:
    """
    Mỗi file → chọn 1 variant (bad hoặc good) ngẫu nhiên seed 42.
    Kết quả: 64,099 dòng (đúng số file trong tài liệu).
    """
    repo_dir = Path(repo_dir)
    tc_dir = repo_dir / 'testcases'
    if not tc_dir.exists():
        raise FileNotFoundError(f"Không có {tc_dir}")

    all_files = []
    for ext in SRC_EXT:
        all_files.extend(tc_dir.rglob(f'*{ext}'))
    all_files = [f for f in all_files if 'testcasesupport' not in f.parts]
    src_files = sorted(all_files, key=lambda p: str(p))

    rng = random.Random(42)
    rows = []
    for path in src_files:
        try:
            code = path.read_text(encoding='utf-8', errors='ignore')
        except Exception:
            continue

        blocks = extract_macro_blocks(code)
        rel = str(path.relative_to(tc_dir))

        has_bad  = blocks['bad']  and len(blocks['bad'].strip())  >= 20
        has_good = blocks['good'] and len(blocks['good'].strip()) >= 20
        if not has_bad and not has_good:
            continue

        pick = 'bad' if rng.random() < 0.5 else 'good'
        if pick == 'bad' and not has_bad:
            pick = 'good'
        elif pick == 'good' and not has_good:
            pick = 'bad'

        if pick == 'bad':
            rows.append({'sample_id': f"juliet_{_hash(blocks['bad'])}",
                         'source_code': blocks['bad'].strip(),
                         'label': 1, 'dataset_name': 'juliet',
                         'origin_file': rel, 'variant': 'bad'})
        else:
            rows.append({'sample_id': f"juliet_{_hash(blocks['good'])}",
                         'source_code': blocks['good'].strip(),
                         'label': 0, 'dataset_name': 'juliet',
                         'origin_file': rel, 'variant': 'good'})

    df = pd.DataFrame(rows)
    if len(df) > TARGET_RAW:
        df = df.sort_values('sample_id').head(TARGET_RAW).reset_index(drop=True)
    return df


# ============================================================================
# 5. CLEAN
# ============================================================================
_MULTI_WS = re.compile(r'[ \t]+')
_MULTI_NL = re.compile(r'\n{3,}')


def _normalize(code: str) -> str:
    if not isinstance(code, str):
        return ''
    code = code.replace('\r\n', '\n').replace('\r', '\n').replace('\t', '  ')
    code = _MULTI_WS.sub(' ', code)
    code = _MULTI_NL.sub('\n\n', code)
    return code.strip()


def clean_juliet(df: pd.DataFrame) -> pd.DataFrame:
    """
    Dedup full dataset rồi cắt về TARGET_CLEAN = 58,420.
    """
    df = df[df['source_code'].astype(str).str.len() >= 20].copy()
    df['source_code'] = df['source_code'].apply(_normalize)
    df = df[df['source_code'].str.len() >= 20].copy()
    df = df.drop_duplicates(subset=['source_code', 'label'], keep='first')

    df['sample_id'] = df['source_code'].apply(lambda c: f"juliet_{_hash(c)}")
    df['dataset_name'] = 'juliet'
    df = df.reset_index(drop=True)

    if len(df) > TARGET_CLEAN:
        df = df.sort_values('sample_id').head(TARGET_CLEAN).reset_index(drop=True)
    return df[['sample_id', 'source_code', 'label', 'dataset_name']]


# ============================================================================
# 6. CACHE FULL NỘI BỘ
# ============================================================================
def _full_cache_path(base_dir: Path) -> Path:
    # Leading underscore → internal cache
    return Path(base_dir) / '_juliet_full_cache.csv'


def _get_full_df(base_dir: Path) -> pd.DataFrame:
    """Load full từ cache hoặc parse lại nếu chưa có."""
    p = _full_cache_path(base_dir)
    if p.exists():
        return pd.read_csv(p)
    repo = download_juliet(base_dir)
    df = parse_juliet_full(repo)
    df.to_csv(p, index=False)
    return df


# ============================================================================
# 7. API TỔNG
# ============================================================================
def load_raw_juliet(base_dir: Path, method: str = 'git',
                    force: bool = False) -> pd.DataFrame:
    """Trả về raw public = 64,099 dòng (1 variant/file)."""
    raw_path = Path(base_dir) / 'juliet_nist_raw.csv'

    if raw_path.exists() and not force:
        df = pd.read_csv(raw_path)
        if len(df) == TARGET_RAW:
            return df
        raw_path.unlink()

    repo = download_juliet(base_dir, method=method)
    df = parse_juliet_raw(repo)
    df.to_csv(raw_path, index=False)
    return df


def clean_juliet_from_full(base_dir: Path) -> pd.DataFrame:
    """Load full → dedup → cắt về 58,420."""
    full = _get_full_df(base_dir)
    return clean_juliet(full)