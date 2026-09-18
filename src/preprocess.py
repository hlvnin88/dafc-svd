"""Bước 2: Tiền xử lý dữ liệu.

Theo Phụ lục A (Bảng A1):
  - Chuẩn hóa UTF-8, LF, khoảng trắng.
  - Loại mẫu rỗng, thiếu nhãn, lỗi định dạng, không đọc được.
  - Loại duplicate trước khi chia train/val/test.
  - Cùng 1 pipeline cho cả 3 dataset.
"""
import re
import pandas as pd

from config import DATA_DIR, DATASET_JULIET
from src.utils import ensure_dir


# ============================================================================
# HẰNG SỐ
# ============================================================================
_MIN_CODE_LEN = 20
_MULTI_WS = re.compile(r'[ \t]+')
_MULTI_NL = re.compile(r'\n{3,}')


# ============================================================================
# NORMALIZE — chuẩn hóa UTF-8, LF, khoảng trắng
# ============================================================================
def normalize_code(code: str) -> str:
    """Chuẩn hóa theo Phụ lục A."""
    if not isinstance(code, str):
        return ''
    # UTF-8 round-trip
    code = code.encode('utf-8', errors='ignore').decode('utf-8', errors='ignore')
    # LF line endings
    code = code.replace('\r\n', '\n').replace('\r', '\n')
    # Tab → 2 space
    code = code.replace('\t', '  ')
    # Gộp khoảng trắng
    code = _MULTI_WS.sub(' ', code)
    # Gộp dòng trống
    code = _MULTI_NL.sub('\n\n', code)
    return code.strip()


# ============================================================================
# VALIDATION — loại rỗng, thiếu nhãn, lỗi định dạng, không đọc được
# ============================================================================
def _is_valid_utf8(code: str) -> bool:
    """Kiểm tra code là UTF-8 hợp lệ."""
    if not isinstance(code, str):
        return False
    try:
        code.encode('utf-8').decode('utf-8')
        return True
    except (UnicodeDecodeError, UnicodeEncodeError):
        return False


def _looks_like_source_code(code: str) -> bool:
    """
    Kiểm tra sơ bộ code giống mã C/C++ (không phải binary/random text).
    Dấu hiệu:
      - Có ít nhất 1 trong các ký tự đặc trưng: { } ( ) ; ,
      - Không chứa quá nhiều ký tự điều khiển (>1% → binary)
    """
    if not code:
        return False
    # Bắt buộc có ít nhất 1 dấu đặc trưng của C/C++
    if not any(ch in code for ch in '{}();'):
        return False
    # Không quá nhiều ký tự điều khiển
    n_control = sum(1 for ch in code if ord(ch) < 9 or (13 < ord(ch) < 32))
    if n_control / max(len(code), 1) > 0.01:
        return False
    return True


def _valid(code, label) -> bool:
    """
    Kiểm tra record hợp lệ:
      - Là string, không rỗng, độ dài >= 20 ký tự
      - Trông giống mã C/C++
      - Label là 0 hoặc 1
    """
    if not isinstance(code, str):
        return False
    if len(code.strip()) < _MIN_CODE_LEN:
        return False
    if not _looks_like_source_code(code):
        return False
    try:
        v = int(label)
    except Exception:
        return False
    if v not in (0, 1):
        return False
    return True


# ============================================================================
# PIPELINE
# ============================================================================
def run_preprocess(name: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    Chạy pipeline tiền xử lý theo Phụ lục A.

    Các bước:
      1. Loại missing label / missing code
      2. Loại code không phải UTF-8 hợp lệ
      3. Chuẩn hóa (UTF-8, LF, whitespace)
      4. Loại mẫu rỗng / sai định dạng / độ dài < 20
      5. Loại duplicate
      6. Chuẩn hóa label
    """
    log = {'input': len(df)}

    # --- 1. Loại missing ---
    df = df.dropna(subset=['source_code', 'label']).copy()
    log['after_dropna'] = len(df)

    # --- 2. Loại code không phải UTF-8 hợp lệ ---
    mask_utf8 = df['source_code'].apply(_is_valid_utf8)
    df = df[mask_utf8].copy()
    log['after_utf8_check'] = len(df)

    # --- 3. Chuẩn hóa ---
    df['source_code'] = df['source_code'].apply(normalize_code)

    # --- 4. Validation: rỗng, sai định dạng, độ dài ---
    mask_valid = df.apply(
        lambda r: _valid(r['source_code'], r['label']), axis=1)
    df = df[mask_valid].reset_index(drop=True)
    log['after_validity'] = len(df)

    # --- 5. Loại duplicate ---
    # Juliet: giữ cả good và bad (cùng code, khác label)
    # Devign / Big-Vul: loại theo source_code
    if name == DATASET_JULIET:
        subset = ['source_code', 'label']
    else:
        subset = ['source_code']
    df = df.drop_duplicates(subset=subset, keep='first').reset_index(drop=True)
    log['after_dedup'] = len(df)

    # --- 6. Chuẩn hóa label ---
    df['label'] = df['label'].astype(int)
    df = df[df['label'].isin([0, 1])].reset_index(drop=True)

    # --- Lưu ---
    df = df[['sample_id', 'source_code', 'label', 'dataset_name']]
    out = ensure_dir(DATA_DIR / name / 'clean') / f'{name}_clean.csv'
    df.to_csv(out, index=False)

    print(f"✅ Clean {name}: {log}")
    return df