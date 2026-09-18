#!/usr/bin/env python
"""
Bước 3: Trích xuất F1–F10.
"""
import argparse, sys, time, pickle
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import torch

from config import (DATA_DIR, DATASETS, EXTRACT_LOGS, RESOURCE_LOGS, TABLES_DIR,
                     print_config, FEATURE_GROUPS, MODELS_DIR, F10_METRIC_NAMES,
                     F1_TFIDF_MAX_FEATURES, F1_NGRAM_RANGE,
                     CODEBERT_NAME, GRAPHCODEBERT_NAME, ENCODER_MAX_LEN, DEVICE)
from src.utils import ensure_dir, print_section, set_seed, _mem_mb

from src.feature_io import (save_raw_f1, save_raw_f2, save_raw_f3,
                             save_raw_f8, save_raw_f9, save_raw_f10)
from src.features import (f1_lexical, f2_slice, f3_ast,
                           f8_codebert, f9_graphcodebert, f10_security)
from src.joern_extractor import JoernExtractor


ENC = lambda: ensure_dir(MODELS_DIR / 'encoders')

ALL_FEATURES  = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10']
TOP7_FEATURES = ['F5', 'F8', 'F9', 'F10']


def features_for(dataset_name):
    """Trả về danh sách nhóm F cần extract cho mỗi dataset."""
    return ALL_FEATURES if dataset_name == 'devign' else TOP7_FEATURES


# ============================================================================
# LOGGING THEO PHỤ LỤC A
# ============================================================================
def _append_csv(df, path):
    """Append vào CSV, tạo mới nếu chưa có."""
    path = Path(path)
    if path.exists():
        df = pd.concat([pd.read_csv(path), df], ignore_index=True)
    df.to_csv(path, index=False)


def _log_extraction(name, group, elapsed_sec,
                    n_input, n_success, err=''):
    """
    Ghi extraction log theo Phụ lục A:
        input_samples  — tổng số mẫu đưa vào
        successful     — số mẫu extract thành công
        failed         — input - success
        failure_rate   — failed / input * 100
    """
    n_failed = max(0, n_input - n_success)
    fail_rate = (n_failed / n_input * 100) if n_input > 0 else 0.0

    row = pd.DataFrame([{
        'dataset': name,
        'group': group,
        'input_samples': int(n_input),
        'successful':    int(n_success),
        'failed':        int(n_failed),
        'failure_rate':  round(fail_rate, 2),
        'time_sec':      round(elapsed_sec, 3),
        'error':         err[:200],
    }])
    _append_csv(row, EXTRACT_LOGS / f'{group}.csv')


def _log_resource(name, group, t_ext, t_inf, dim, mem, fail_rate):
    """Ghi resource log cho bảng Cost."""
    row = pd.DataFrame([{
        'dataset': name, 'group': group,
        'extraction_time': t_ext,
        'inference_time':  t_inf,
        'dimension':       dim,
        'memory_mb':       mem,
        'failure_rate':    fail_rate,
    }])
    _append_csv(row, RESOURCE_LOGS / f'{group}.csv')


# ============================================================================
# ĐẾM MẪU EXTRACT THÀNH CÔNG
# ============================================================================
def _count_successful(group, name, split):
    """
    Đếm số mẫu extract thành công = số vector không phải toàn 0.

    Lý do: pipeline dùng mẫu fail → vector 0 để đánh dấu;
           `_valid_intersection` lọc mẫu toàn 0 ở các bước sau.
    """
    p = DATA_DIR / name / 'features' / group / f'{name}_{split}_{group.lower()}.npy'
    if not p.exists():
        return 0
    try:
        arr = np.load(p)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        return int(np.sum(np.any(arr != 0, axis=1)))
    except Exception:
        return 0


def _count_successful_all_splits(group, name, splits=('train', 'val', 'test')):
    """Tổng số mẫu thành công qua các split."""
    return sum(_count_successful(group, name, s) for s in splits)


# ============================================================================
# LOAD SPLIT
# ============================================================================
def _load_splits(name):
    clean = pd.read_csv(DATA_DIR / name / 'clean' / f'{name}_clean.csv')

    def _ids(s):
        p = DATA_DIR / name / 'splits' / f'{name}_{s}_ids.txt'
        if not p.exists():
            return []
        with open(p) as f:
            return [l.strip() for l in f if l.strip()]

    tr = clean[clean['sample_id'].isin(_ids('train'))].reset_index(drop=True)
    va = clean[clean['sample_id'].isin(_ids('val'))].reset_index(drop=True)
    te = clean[clean['sample_id'].isin(_ids('test'))].reset_index(drop=True)
    return clean, tr, va, te


def _save_encoded(Z, name, split, group):
    d = ensure_dir(DATA_DIR / name / 'features' / group)
    np.save(d / f'{name}_{split}_{group.lower()}.npy', Z)


# ============================================================================
# FIT ENCODERS (chỉ Devign)
# ============================================================================
def fit_all_encoders(tr, va, name):
    print_section("FIT ENCODERS (Devign)", '-')
    tr_codes = tr['source_code'].astype(str).tolist()
    va_codes = va['source_code'].astype(str).tolist()
    y_train = tr['label'].values.astype(np.float32)

    # F1: TF-IDF + Linear
    from sklearn.feature_extraction.text import TfidfVectorizer
    print("  ⏳ Fit F1 (TF-IDF 5000 → Linear 256)...")
    vec = TfidfVectorizer(max_features=F1_TFIDF_MAX_FEATURES,
                           ngram_range=F1_NGRAM_RANGE,
                           token_pattern=r'\S+', lowercase=False)
    X_tr = vec.fit_transform(tr_codes)
    with open(ENC() / 'devign_f1_tfidf.pkl', 'wb') as f:
        pickle.dump(vec, f)
    f1_lexical.fit_f1_proj(X_tr.toarray().astype(np.float32), y_train)

    # F2: BiLSTM
    print("  ⏳ Fit F2 (BiLSTM)...")
    f2_slice.fit_f2_bilstm(tr, va, name)

    # F3–F7: GCN
    for group, edge_type in [('F3', 'AST'), ('F4', 'CFG'), ('F5', 'DDG'),
                              ('F6', 'CALL'), ('F7', 'ALL')]:
        print(f"  ⏳ Fit {group} (GCN, edge={edge_type})...")
        f3_ast.fit_gcn_for_edge(tr, name, edge_type, group)

    # F8/F9: Linear 768→256
    from transformers import AutoTokenizer, AutoModel

    def _embed(codes, model_name):
        tok = AutoTokenizer.from_pretrained(model_name)
        mdl = AutoModel.from_pretrained(model_name).eval().to(DEVICE)
        out = []
        with torch.no_grad():
            for i in range(0, len(codes), 16):
                enc = tok(codes[i:i+16], return_tensors='pt',
                          truncation=True, max_length=ENCODER_MAX_LEN,
                          padding=True)
                enc = {k: v.to(DEVICE) for k, v in enc.items()}
                h = mdl(**enc).last_hidden_state.mean(1)
                out.append(h.cpu().numpy().astype(np.float32))
        return np.concatenate(out, 0)

    print("  ⏳ Fit F8 (CodeBERT Linear 768→256)...")
    f8_codebert.fit_f8_proj(_embed(tr_codes, CODEBERT_NAME), y_train)
    print("  ⏳ Fit F9 (GraphCodeBERT Linear 768→256)...")
    f9_graphcodebert.fit_f9_proj(_embed(tr_codes, GRAPHCODEBERT_NAME), y_train)

    # F10: Linear 32→256
    from src.feature_io import _f10_metrics
    X10 = np.stack([[_f10_metrics(c)[m] for m in F10_METRIC_NAMES]
                     for c in tr_codes]).astype(np.float32)
    print("  ⏳ Fit F10 (32 → Linear 256)...")
    f10_security.fit_f10_proj(X10, y_train)

    print("  ✅ Tất cả encoder đã fit xong.")


# ============================================================================
# TRANSFORM + LOG THEO PHỤ LỤC A
# ============================================================================
def transform_subset(name, clean, tr, va, te, needed):
    print_section(f"TRANSFORM — {name} — needed={needed}", '-')
    tr_codes = tr['source_code'].astype(str).tolist()
    va_codes = va['source_code'].astype(str).tolist()
    te_codes = te['source_code'].astype(str).tolist()
    n_input = len(tr_codes) + len(va_codes) + len(te_codes)

    # ========================================================================
    # RAW ARTIFACTS
    # ========================================================================
    print_section("   RAW artifacts", '.')

    # F1
    if 'F1' in needed:
        vec = None
        if name != 'devign':
            with open(ENC() / 'devign_f1_tfidf.pkl', 'rb') as f:
                vec = pickle.load(f)
        _, vec = save_raw_f1(tr, va, te, name, vectorizer=vec)
        with open(ENC() / 'devign_f1_tfidf.pkl', 'wb') as f:
            pickle.dump(vec, f)

    # F2
    if 'F2' in needed:
        save_raw_f2(tr, va, te, name)

    # F3–F7
    if 'F3' in needed: save_raw_f3(clean, name, edge_type='AST')
    if 'F4' in needed: save_raw_f3(clean, name, edge_type='CFG')
    if 'F5' in needed: save_raw_f3(clean, name, edge_type='DDG')
    if 'F6' in needed: save_raw_f3(clean, name, edge_type='CALL')
    if 'F7' in needed: save_raw_f3(clean, name, edge_type='ALL')

    # F8/F9
    emb8 = emb9 = None
    if 'F8' in needed:
        _, emb8 = save_raw_f8(tr_codes, va_codes, te_codes, name)
    if 'F9' in needed:
        _, emb9 = save_raw_f9(tr_codes, va_codes, te_codes, name)

    # F10
    X10 = None
    if 'F10' in needed:
        _, X10 = save_raw_f10(tr, va, te, name)

    # ========================================================================
    # ENCODED 256-dim — có log theo Phụ lục A
    # ========================================================================
    print_section("   Encoded 256-dim", '.')

    # --- F1 ---
    if 'F1' in needed:
        t0 = time.time(); m0 = _mem_mb()
        try:
            with open(ENC() / 'devign_f1_tfidf.pkl', 'rb') as f:
                vec = pickle.load(f)
            X_tr = vec.transform(tr_codes).toarray().astype(np.float32)
            X_va = vec.transform(va_codes).toarray().astype(np.float32)
            X_te = vec.transform(te_codes).toarray().astype(np.float32)
            proj = f1_lexical.load_f1_proj(X_tr.shape[1])
            Z_tr = f1_lexical.apply_f1_proj(proj, X_tr)
            Z_va = f1_lexical.apply_f1_proj(proj, X_va)
            Z_te = f1_lexical.apply_f1_proj(proj, X_te)
            _save_encoded(Z_tr, name, 'train', 'F1')
            _save_encoded(Z_va, name, 'val',   'F1')
            _save_encoded(Z_te, name, 'test',  'F1')
            n_succ = _count_successful_all_splits('F1', name)
            _log_extraction(name, 'F1', time.time() - t0, n_input, n_succ)
            _log_resource(name, 'F1', time.time() - t0, 0.0, 5000,
                          _mem_mb() - m0, 0.0)
            print(f"  ✅ F1 encoded ({n_succ}/{n_input})")
        except Exception as e:
            _log_extraction(name, 'F1', time.time() - t0, n_input, 0, str(e))
            print(f"  ❌ F1: {e}")

    # --- F2 ---
    if 'F2' in needed:
        t0 = time.time(); m0 = _mem_mb()
        try:
            m2, v2 = f2_slice.load_f2_bilstm()
            for split_df, split_name in [(tr, 'train'), (va, 'val'), (te, 'test')]:
                if len(split_df) == 0:
                    continue
                s = split_df.copy()
                s.attrs['dataset_name'] = name
                _save_encoded(f2_slice.apply_f2(m2, v2, s), name, split_name, 'F2')
            n_succ = _count_successful_all_splits('F2', name)
            _log_extraction(name, 'F2', time.time() - t0, n_input, n_succ)
            _log_resource(name, 'F2', time.time() - t0, 0.0, 256,
                          _mem_mb() - m0, 0.0)
            print(f"  ✅ F2 encoded ({n_succ}/{n_input})")
        except Exception as e:
            _log_extraction(name, 'F2', time.time() - t0, n_input, 0, str(e))
            print(f"  ❌ F2: {e}")

    # --- F3–F7 ---
    for group, edge_type in [('F3', 'AST'), ('F4', 'CFG'), ('F5', 'DDG'),
                              ('F6', 'CALL'), ('F7', 'ALL')]:
        if group not in needed:
            continue
        t0 = time.time(); m0 = _mem_mb()
        try:
            m, v = f3_ast.load_gcn_for_edge(group)
            for split_df, split_name in [(tr, 'train'), (va, 'val'), (te, 'test')]:
                if len(split_df) == 0:
                    continue
                _save_encoded(
                    f3_ast.apply_gcn(m, v, split_df, name, edge_type),
                    name, split_name, group)
            n_succ = _count_successful_all_splits(group, name)
            _log_extraction(name, group, time.time() - t0, n_input, n_succ)
            _log_resource(name, group, time.time() - t0, 0.0, 256,
                          _mem_mb() - m0, 0.0)
            print(f"  ✅ {group} encoded ({n_succ}/{n_input})")
        except Exception as e:
            _log_extraction(name, group, time.time() - t0, n_input, 0, str(e))
            print(f"  ❌ {group}: {e}")

    # --- F8 ---
    if 'F8' in needed and emb8 is not None:
        t0 = time.time(); m0 = _mem_mb()
        try:
            n_tr, n_va, n_te = len(tr), len(va), len(te)
            e_tr = emb8[:n_tr]
            e_va = emb8[n_tr:n_tr+n_va]
            e_te = emb8[n_tr+n_va:]
            proj = f8_codebert.load_f8_proj()
            if n_tr: _save_encoded(f8_codebert.apply_f8(proj, e_tr), name, 'train', 'F8')
            if n_va: _save_encoded(f8_codebert.apply_f8(proj, e_va), name, 'val',   'F8')
            if n_te: _save_encoded(f8_codebert.apply_f8(proj, e_te), name, 'test',  'F8')
            n_succ = _count_successful_all_splits('F8', name)
            _log_extraction(name, 'F8', time.time() - t0, n_input, n_succ)
            _log_resource(name, 'F8', time.time() - t0, 0.0, 768,
                          _mem_mb() - m0, 0.0)
            print(f"  ✅ F8 encoded ({n_succ}/{n_input})")
        except Exception as e:
            _log_extraction(name, 'F8', time.time() - t0, n_input, 0, str(e))
            print(f"  ❌ F8: {e}")

    # --- F9 ---
    if 'F9' in needed and emb9 is not None:
        t0 = time.time(); m0 = _mem_mb()
        try:
            n_tr, n_va, n_te = len(tr), len(va), len(te)
            e_tr = emb9[:n_tr]
            e_va = emb9[n_tr:n_tr+n_va]
            e_te = emb9[n_tr+n_va:]
            proj = f9_graphcodebert.load_f9_proj()
            if n_tr: _save_encoded(f9_graphcodebert.apply_f9(proj, e_tr), name, 'train', 'F9')
            if n_va: _save_encoded(f9_graphcodebert.apply_f9(proj, e_va), name, 'val',   'F9')
            if n_te: _save_encoded(f9_graphcodebert.apply_f9(proj, e_te), name, 'test',  'F9')
            n_succ = _count_successful_all_splits('F9', name)
            _log_extraction(name, 'F9', time.time() - t0, n_input, n_succ)
            _log_resource(name, 'F9', time.time() - t0, 0.0, 768,
                          _mem_mb() - m0, 0.0)
            print(f"  ✅ F9 encoded ({n_succ}/{n_input})")
        except Exception as e:
            _log_extraction(name, 'F9', time.time() - t0, n_input, 0, str(e))
            print(f"  ❌ F9: {e}")

    # --- F10 ---
    if 'F10' in needed and X10 is not None:
        t0 = time.time(); m0 = _mem_mb()
        try:
            n_tr, n_va, n_te = len(tr), len(va), len(te)
            X10_tr = X10[:n_tr]
            X10_va = X10[n_tr:n_tr+n_va]
            X10_te = X10[n_tr+n_va:]
            proj = f10_security.load_f10_proj(X10.shape[1])
            if n_tr: _save_encoded(f10_security.apply_f10(proj, X10_tr), name, 'train', 'F10')
            if n_va: _save_encoded(f10_security.apply_f10(proj, X10_va), name, 'val',   'F10')
            if n_te: _save_encoded(f10_security.apply_f10(proj, X10_te), name, 'test',  'F10')
            n_succ = _count_successful_all_splits('F10', name)
            _log_extraction(name, 'F10', time.time() - t0, n_input, n_succ)
            _log_resource(name, 'F10', time.time() - t0, 0.0, 32,
                          _mem_mb() - m0, 0.0)
            print(f"  ✅ F10 encoded ({n_succ}/{n_input})")
        except Exception as e:
            _log_extraction(name, 'F10', time.time() - t0, n_input, 0, str(e))
            print(f"  ❌ F10: {e}")


# ============================================================================
# MAIN
# ============================================================================
def extract_dataset(name):
    print_section(f"BƯỚC 3 — {name}", '=')
    clean, tr, va, te = _load_splits(name)
    print(f"  clean={len(clean)} train={len(tr)} val={len(va)} test={len(te)}")

    needed = features_for(name)
    print(f"  Features to extract: {needed}")
    print(f"  skip Joern slice: {'No' if name == 'devign' else 'Yes'}")

    # Joern cache — bỏ slice nếu không cần F2
    jc = DATA_DIR / name / 'joern_raw'
    jc.mkdir(parents=True, exist_ok=True)
    need_joern = any(g in needed for g in ['F3', 'F4', 'F5', 'F6', 'F7'])
    need_slice = 'F2' in needed
    if need_joern and len(list(jc.glob('*.json'))) < len(clean) * 0.5:
        try:
            print(f"⏳ Chạy Joern (skip_slice={not need_slice})...")
            JoernExtractor().extract_dataset(clean, name,
                                              skip_slice=not need_slice)
        except Exception as e:
            print(f"⚠️ Joern fail: {e}")

    # Fit encoder (chỉ Devign)
    if name == 'devign':
        fit_all_encoders(tr, va, name)

    # Transform subset
    transform_subset(name, clean, tr, va, te, needed)


def main(dataset):
    print_config()
    set_seed(42)

    if dataset == 'all':
        for t in ['devign', 'juliet', 'bigvul']:
            extract_dataset(t)
    else:
        extract_dataset(dataset)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='all',
                    choices=['devign', 'juliet', 'bigvul', 'all'])
    main(ap.parse_args().dataset)