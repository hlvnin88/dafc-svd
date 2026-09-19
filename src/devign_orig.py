"""
Tải và xử lý bộ dữ liệu Devign gốc (Zhou et al., NeurIPS 2019).

Nguồn function.json:
https://drive.google.com/file/d/1x6hoF7G-tSYxg8AFybggypLZgMGDNHfF

Giữ nguyên cách parse, chuẩn hóa và làm sạch của repository DAFC-SVD.
Phần tải hỗ trợ cả nội dung JSON trực tiếp và tệp ZIP.
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
DEVIGN_ZIP_NAME = "devign_function.json.zip"
DEVIGN_JSON_NAME = "function.json"

TARGET_RAW = 27318
TARGET_CLEAN = 24754


# ============================================================================
# 1. TẢI
# ============================================================================
def download_devign(base_dir: Path, force: bool = False) -> Path:
    base_dir = Path(base_dir)
    ensure_dir(base_dir)
    json_path = base_dir / DEVIGN_JSON_NAME

    if json_path.is_file() and not force:
        return json_path

    try:
        import gdown
    except ImportError as exc:
        raise RuntimeError("Cần cài thư viện gdown: pip install gdown") from exc

    out = base_dir / DEVIGN_ZIP_NAME
    temp_json = base_dir / "function.json.part"
    last_error = None

    urls = [
        None,
        f"https://drive.google.com/uc?id={DEVIGN_GDRIVE_ID}",
        f"https://drive.google.com/uc?export=download&id={DEVIGN_GDRIVE_ID}",
    ]

    for url in urls:
        try:
            # Loại bỏ file tải dở từ lần thử trước (không xóa JSON hợp lệ).
            if out.exists():
                out.unlink()
            if temp_json.exists():
                temp_json.unlink()

            if url is None:
                result = gdown.download(
                    id=DEVIGN_GDRIVE_ID,
                    output=str(out),
                    quiet=True,
                )
            else:
                result = gdown.download(
                    url,
                    output=str(out),
                    quiet=True,
                )

            if not result or not out.is_file() or out.stat().st_size == 0:
                raise RuntimeError("gdown không trả về file tải hợp lệ")

            if zipfile.is_zipfile(out):
                # ZIP: chỉ lấy file JSON cần thiết, không giải nén tùy ý.
                with zipfile.ZipFile(out, "r") as archive:
                    members = [
                        info for info in archive.infolist()
                        if not info.is_dir()
                        and info.filename.lower().endswith(".json")
                    ]
                    if not members:
                        raise RuntimeError("ZIP không chứa file JSON")
                    exact = [
                        info for info in members
                        if Path(info.filename).name == DEVIGN_JSON_NAME
                    ]
                    # Ưu tiên function.json; nếu không có, lấy file JSON lớn nhất.
                    member = max(exact or members, key=lambda x: x.file_size)
                    with archive.open(member, "r") as source, temp_json.open("wb") as target:
                        shutil.copyfileobj(source, target)
            else:
                # Google Drive có thể trả JSON dù tên file đầu ra mang đuôi ZIP.
                with out.open("rb") as source:
                    header = source.read(256).lstrip()
                if not header.startswith((b"[", b"{")):
                    raise RuntimeError(
                        "File tải về không phải ZIP hoặc JSON; "
                        "hãy kiểm tra quyền truy cập Google Drive"
                    )
                shutil.move(str(out), str(temp_json))

            # Kiểm tra sơ bộ file thu được trước khi thay file gốc.
            with temp_json.open("rb") as source:
                header = source.read(256).lstrip()
            if not header.startswith((b"[", b"{")):
                raise RuntimeError("Nội dung JSON sau tải/giải nén không hợp lệ")

            temp_json.replace(json_path)
            if out.exists():
                out.unlink()
            return json_path

        except Exception as exc:
            last_error = exc
            if temp_json.exists():
                temp_json.unlink()
            if out.exists():
                out.unlink()

    raise RuntimeError(f"Tải Devign thất bại: {last_error}") from last_error


# ============================================================================
# 2. PARSE
# ============================================================================
def parse_devign(json_path: Path) -> pd.DataFrame:
    json_path = Path(json_path)
    rows = []
    with open(json_path, "r", encoding="utf-8", errors="ignore") as f:
        head = f.read(200).lstrip()
        f.seek(0)
        if head.startswith("["):
            for item in json.load(f):
                rows.append(_obj_to_row(item))
        else:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(_obj_to_row(json.loads(line)))
                except json.JSONDecodeError:
                    continue
    return pd.DataFrame(rows)


def _obj_to_row(obj: dict) -> dict:
    func = obj.get("func", "") or obj.get("function", "") or ""
    target = obj.get("target", obj.get("label", 0))
    try:
        target = int(target)
    except (TypeError, ValueError):
        target = 0
    return {
        "source_code": str(func),
        "label": target,
        "project": str(obj.get("project", "")),
        "commit_id": str(obj.get("commit_id", "")),
    }


# ============================================================================
# 3. NORMALIZE (giữ nguyên logic gốc của repository)
# ============================================================================
_STR_LITERAL = re.compile(r'"(?:[^"\\]|\\.)*"')
_CHR_LITERAL = re.compile(r"'(?:[^'\\]|\\.)*'")
_LINE_COMMENT = re.compile(r'//[^\n]*')
_BLOCK_COMMENT = re.compile(r'/\*.*?\*/', re.DOTALL)
_NUMBER = re.compile(r'\b\d+(?:\.\d+)?\b')
_IDENT = re.compile(r'\b[A-Za-z_]\w*\b')
_WS = re.compile(r'\s+')

_TYPE_KEYWORDS = {
    'int', 'char', 'float', 'double', 'long', 'short', 'void', 'signed', 'unsigned',
    'const', 'static', 'volatile', 'inline', 'register', 'auto', 'extern',
    'struct', 'union', 'enum', '_Bool', 'bool', 'size_t', 'ssize_t',
}
_CTRL_KEYWORDS = {
    'if', 'else', 'for', 'while', 'do', 'switch', 'case', 'default', 'break',
    'continue', 'return', 'goto', 'sizeof', 'typedef',
    'class', 'namespace', 'template', 'typename', 'public', 'private', 'protected',
    'virtual', 'override', 'new', 'delete', 'nullptr', 'using', 'true', 'false',
    'try', 'catch', 'throw', 'this', 'operator',
}


def _normalize_for_dedup(code: str) -> str:
    if not isinstance(code, str):
        return ''
    code = _LINE_COMMENT.sub('', code)
    code = _BLOCK_COMMENT.sub('', code)
    code = _STR_LITERAL.sub('"S"', code)
    code = _CHR_LITERAL.sub("'C'", code)
    code = _NUMBER.sub('0', code)

    def repl(m):
        w = m.group(0)
        if w in _TYPE_KEYWORDS:
            return 'T'
        if w in _CTRL_KEYWORDS:
            return w
        return 'V'

    code = _IDENT.sub(repl, code)
    code = _WS.sub(' ', code).strip()
    return code


# ============================================================================
# 4. CLEAN (giữ nguyên tiêu chí và giới hạn mẫu của repository)
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
        lambda c: f"devign_{hashlib.md5(c.encode('utf-8', errors='ignore')).hexdigest()[:12]}"
    )
    df['dataset_name'] = 'devign'

    if len(df) > TARGET_CLEAN:
        df = df.sort_values('sample_id').head(TARGET_CLEAN).reset_index(drop=True)
    return df[['sample_id', 'source_code', 'label', 'dataset_name']]


# ============================================================================
# 5. API TỔNG
# ============================================================================
def load_raw_devign(base_dir: Path, force: bool = False) -> pd.DataFrame:
    base_dir = Path(base_dir)
    ensure_dir(base_dir)
    raw_path = base_dir / 'devign_raw.csv'

    if raw_path.exists() and not force:
        return pd.read_csv(raw_path)

    json_path = download_devign(base_dir, force=force)
    df = parse_devign(json_path)
    df.to_csv(raw_path, index=False)
    return df
