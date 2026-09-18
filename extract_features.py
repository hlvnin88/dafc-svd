#!/usr/bin/env python
"""
Bước 3: Sinh 10 file raw artifact (Bảng 4) + encoded 256-dim.

Tối ưu:
    Devign  → F1..F10  (đầy đủ, cần cho individual + 375 combos)
    Juliet  → F5, F8, F9, F10  (hợp của Top-7)
    Big-Vul → F5, F8, F9, F10  (hợp của Top-7)
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


# ============================================================================
# NHÓM F CẦN EXTRACT THEO DATASET
# ============================================================================
ALL_FEATURES    = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10']
TOP7_FEATURES   = ['F5', 'F8', 'F9', 'F10']   # hợp của Top-7 (S1..S7)

def features_for(dataset_name):
    return ALL_FEATURES if dataset_name == 'devign' else TOP7_FEATURES


# ============================================================================
# BẢNG #2 — feature_config.csv
# ============================================================================
def write_feature_config():
    rows = [
        {'Nhóm': 'F1',  'Nội dung biểu diễn': 'Lexical/Text',
         'Bộ trích xuất / Encoder': 'TF-IDF trên token và n-gram',
         'Kích thước biểu diễn': '5,000, ánh xạ xuống 256'},
        {'Nhóm': 'F2',  'Nội dung biểu diễn': 'Sequence/Slice',
         'Bộ trích xuất / Encoder': 'Program slice + BiLSTM',
         'Kích thước biểu diễn': '256'},
        {'Nhóm': 'F3',  'Nội dung biểu diễn': 'Syntax/AST',
         'Bộ trích xuất / Encoder': 'Joern AST + GCN',
         'Kích thước biểu diễn': '256'},
        {'Nhóm': 'F4',  'Nội dung biểu diễn': 'Control-flow',
         'Bộ trích xuất / Encoder': 'Joern CFG + GCN',
         'Kích thước biểu diễn': '256'},
        {'Nhóm': 'F5',  'Nội dung biểu diễn': 'Data-flow/Dependency',
         'Bộ trích xuất / Encoder': 'Joern data-flow/dependency graph + GCN',
         'Kích thước biểu diễn': '256'},
        {'Nhóm': 'F6',  'Nội dung biểu diễn': 'Call/Inter-procedural',
         'Bộ trích xuất / Encoder': 'Joern call graph + GCN',
         'Kích thước biểu diễn': '256'},
        {'Nhóm': 'F7',  'Nội dung biểu diễn': 'Composite graph',
         'Bộ trích xuất / Encoder': 'Joern Code Property Graph + GCN',
         'Kích thước biểu diễn': '256'},
        {'Nhóm': 'F8',  'Nội dung biểu diễn': 'Deep semantic embedding',
         'Bộ trích xuất / Encoder': 'CodeBERT-base',
         'Kích thước biểu diễn': '768, ánh xạ xuống 256'},
        {'Nhóm': 'F9',  'Nội dung biểu diễn': 'Deep structural embedding',
         'Bộ trích xuất / Encoder': 'GraphCodeBERT-base',
         'Kích thước biểu diễn': '768, ánh xạ xuống 256'},
        {'Nhóm': 'F10', 'Nội dung biểu diễn': 'Security-risk metrics',
         'Bộ trích xuất / Encoder': '32 static security metrics',
         'Kích thước biểu diễn': '32, ánh xạ lên 256'},
    ]
    df = pd.DataFrame(rows)
    p = TABLES_DIR / 'feature_config.csv'
    df.to_csv(p, index=False)
    print(f"  💾 {p}")


# ============================================================================
# BẢNG #3 — extraction.csv
# ============================================================================
FEATURE_FILE_NAMES = {
    'F1':  'F1_lexical_text.csv',
    'F2':  'F2_sequence_slice.jsonl',
    'F3':  'F3_ast.jsonl',
    'F4':  'F4_cfg.pt',
    'F5':  'F5_dataflow_dependency.pt',
    'F6':  'F6_call_interprocedural.pt',
    'F7':  'F7_composite_graph.pt',
    'F8':  'F8_deep_semantic.npy',
    'F9':  'F9_deep_structural.npy',
    'F10': 'F10_security_metrics.csv',
}

def _fmt_time(sec):
    return f"{int(sec // 60)} min" if sec < 3600 else f"{sec / 3600:.2f} h"

def _dir_size_gb(path):
    if not path.exists(): return 0.0
    return sum(f.stat().st_size for f in path.rglob('*') if f.is_file()) / (1024**3)

def write_extraction_table(dataset_name='devign'):
    rows = []
    total_input = total_succ = total_fail = 0
    for g in FEATURE_GROUPS:
        ext_log = EXTRACT_LOGS / f'{g}.csv'
        res_log = RESOURCE_LOGS / f'{g}.csv'
        if not ext_log.exists(): continue
        df_e = pd.read_csv(ext_log)
        df_e = df_e[df_e['dataset'] == dataset_name]
        if df_e.empty: continue
        n_input = int(df_e['n_samples'].iloc[-1])
        fail_pct = 0.0
        if res_log.exists():
            df_r = pd.read_csv(res_log)
            df_r = df_r[df_r['dataset'] == dataset_name]
            if not df_r.empty:
                fail_pct = float(df_r['failure_rate'].iloc[-1])
        n_fail = int(round(n_input * fail_pct / 100.0))
        n_succ = n_input - n_fail
        cov = round(n_succ / n_input * 100, 2) if n_input else 0.0
        t_str = _fmt_time(float(df_e['time_sec'].iloc[-1]))
        storage = _dir_size_gb(DATA_DIR / dataset_name / 'features' / g)
        rows.append({
            'Group': g, 'Feature file': FEATURE_FILE_NAMES[g],
            'Input samples': f'{n_input:,}', 'Successful': f'{n_succ:,}',
            'Failed': f'{n_fail:,}', 'Coverage (%)': f'{cov:.2f}',
            'Extraction time': t_str, 'Storage size': f'{storage:.2f} GB',
        })
        total_input += n_input; total_succ += n_succ; total_fail += n_fail
    df = pd.DataFrame(rows)
    out = TABLES_DIR / 'extraction.csv'
    df.to_csv(out, index=False)
    avg = total_succ / total_input * 100 if total_input else 0
    print(f"  💾 {out}  |  total={total_input:,}  succ={total_succ:,}  "
          f"fail={total_fail:,}  avg={avg:.2f}%")
    return df


# ============================================================================
# LOAD SPLIT
# ============================================================================
def _load_splits(name):
    clean = pd.read_csv(DATA_DIR / name / 'clean' / f'{name}_clean.csv')
    def _ids(s):
        p = DATA_DIR / name / 'splits' / f'{name}_{s}_ids.txt'
        if not p.exists(): return []
        with open(p) as f: return [l.strip() for l in f if l.strip()]
    tr = clean[clean['sample_id'].isin(_ids('train'))].reset_index(drop=True)
    va = clean[clean['sample_id'].isin(_ids('val'))].reset_index(drop=True)
    te = clean[clean['sample_id'].isin(_ids('test'))].reset_index(drop=True)
    return clean, tr, va, te


def _save_encoded(Z, name, split, group):
    d = ensure_dir(DATA_DIR / name / 'features' / group)
    np.save(d / f'{name}_{split}_{group.lower()}.npy', Z)


def _has_split(df):
    return len(df) > 0


# ============================================================================
# FIT ENCODER
# ============================================================================
def fit_all_encoders(tr, va, name):
    print_section("FIT ENCODERS (Devign)", '-')
    tr_codes = tr['source_code'].astype(str).tolist()
    va_codes = va['source_code'].astype(str).tolist()
    y_train  = tr['label'].values.astype(np.float32)

    # F1
    from sklearn.feature_extraction.text import TfidfVectorizer
    print("  ⏳ Fit F1 (TF-IDF 5000 → Linear 256)...")
    vec = TfidfVectorizer(max_features=F1_TFIDF_MAX_FEATURES,
                           ngram_range=F1_NGRAM_RANGE,
                           token_pattern=r'\S+', lowercase=False)
    X_tr = vec.fit_transform(tr_codes)
    with open(ENC() / 'devign_f1_tfidf.pkl', 'wb') as f:
        pickle.dump(vec, f)
    f1_lexical.fit_f1_proj(X_tr.toarray().astype(np.float32), y_train)

    # F2
    print("  ⏳ Fit F2 (BiLSTM)...")
    f2_slice.fit_f2_bilstm(tr, va, name)

    # F3–F7 GCN
    for group, edge_type in [('F3','AST'),('F4','CFG'),('F5','DDG'),
                              ('F6','CALL'),('F7','ALL')]:
        print(f"  ⏳ Fit {group} (GCN, edge={edge_type})...")
        f3_ast.fit_gcn_for_edge(tr, name, edge_type, group)

    # F8/F9
    from transformers import AutoTokenizer, AutoModel
    def _embed(codes, model_name):
        tok = AutoTokenizer.from_pretrained(model_name)
        mdl = AutoModel.from_pretrained(model_name).eval().to(DEVICE)
        out = []
        with torch.no_grad():
            for i in range(0, len(codes), 16):
                enc = tok(codes[i:i+16], return_tensors='pt', truncation=True,
                          max_length=ENCODER_MAX_LEN, padding=True)
                enc = {k: v.to(DEVICE) for k, v in enc.items()}
                out.append(mdl(**enc).last_hidden_state.mean(1).cpu().numpy().astype(np.float32))
        return np.concatenate(out, 0)

    print("  ⏳ Fit F8 (CodeBERT Linear 768→256)...")
    f8_codebert.fit_f8_proj(_embed(tr_codes, CODEBERT_NAME), y_train)
    print("  ⏳ Fit F9 (GraphCodeBERT Linear 768→256)...")
    f9_graphcodebert.fit_f9_proj(_embed(tr_codes, GRAPHCODEBERT_NAME), y_train)

    # F10
    from src.feature_io import _f10_metrics
    X10 = np.stack([[_f10_metrics(c)[m] for m in F10_METRIC_NAMES]
                     for c in tr_codes]).astype(np.float32)
    print("  ⏳ Fit F10 (32 → Linear 256)...")
    f10_security.fit_f10_proj(X10, y_train)
    print("  ✅ Tất cả encoder đã fit xong.")


# ============================================================================
# TRANSFORM THEO SUBSET
# ============================================================================
def transform_subset(name, clean, tr, va, te, needed):
    """Chỉ xử lý các nhóm F có trong `needed`."""
    print_section(f"TRANSFORM — {name} — needed={needed}", '-')
    tr_codes = tr['source_code'].astype(str).tolist()
    va_codes = va['source_code'].astype(str).tolist()
    te_codes = te['source_code'].astype(str).tolist()

    # ==========================================================
    # RAW ARTIFACTS
    # ==========================================================
    print_section("   RAW artifacts", '.')

    # F1 — chỉ Devign
    if 'F1' in needed:
        vec = None
        if name != 'devign':
            with open(ENC() / 'devign_f1_tfidf.pkl', 'rb') as f:
                vec = pickle.load(f)
        _, vec = save_raw_f1(tr, va, te, name, vectorizer=vec)
        with open(ENC() / 'devign_f1_tfidf.pkl', 'wb') as f:
            pickle.dump(vec, f)

    # F2 — chỉ Devign
    if 'F2' in needed:
        save_raw_f2(tr, va, te, name)

    # F3/F4/F5/F6/F7 — Joern-based
    if 'F3' in needed:
        save_raw_f3(clean, name, edge_type='AST')
    if 'F4' in needed:
        save_raw_f3(clean, name, edge_type='CFG')
    if 'F5' in needed:
        save_raw_f3(clean, name, edge_type='DDG')
    if 'F6' in needed:
        save_raw_f3(clean, name, edge_type='CALL')
    if 'F7' in needed:
        save_raw_f3(clean, name, edge_type='ALL')

    # F8/F9 — CodeBERT
    emb8 = emb9 = None
    if 'F8' in needed:
        _, emb8 = save_raw_f8(tr_codes, va_codes, te_codes, name)
    if 'F9' in needed:
        _, emb9 = save_raw_f9(tr_codes, va_codes, te_codes, name)

    # F10
    X10 = None
    if 'F10' in needed:
        _, X10 = save_raw_f10(tr, va, te, name)

    # ==========================================================
    # ENCODED 256-dim
    # ==========================================================
    print_section("   Encoded 256-dim", '.')

    # F1
    if 'F1' in needed:
        with open(ENC() / 'devign_f1_tfidf.pkl', 'rb') as f:
            vec = pickle.load(f)
        X_tr = vec.transform(tr_codes).toarray().astype(np.float32)
        X_va = vec.transform(va_codes).toarray().astype(np.float32)
        X_te = vec.transform(te_codes).toarray().astype(np.float32)
        proj1 = f1_lexical.load_f1_proj(X_tr.shape[1])
        _save_encoded(f1_lexical.apply_f1_proj(proj1, X_tr), name, 'train', 'F1')
        _save_encoded(f1_lexical.apply_f1_proj(proj1, X_va), name, 'val',   'F1')
        _save_encoded(f1_lexical.apply_f1_proj(proj1, X_te), name, 'test',  'F1')
        print("  ✅ F1 encoded")

    # F2
    if 'F2' in needed:
        m2, v2 = f2_slice.load_f2_bilstm()
        for split_df, split_name in [(tr,'train'),(va,'val'),(te,'test')]:
            if len(split_df) == 0: continue
            s = split_df.copy(); s.attrs['dataset_name'] = name
            _save_encoded(f2_slice.apply_f2(m2, v2, s), name, split_name, 'F2')
        print("  ✅ F2 encoded")

    # F3–F7
    for group, edge_type in [('F3','AST'),('F4','CFG'),('F5','DDG'),
                              ('F6','CALL'),('F7','ALL')]:
        if group not in needed: continue
        m, v = f3_ast.load_gcn_for_edge(group)
        for split_df, split_name in [(tr,'train'),(va,'val'),(te,'test')]:
            if len(split_df) == 0: continue
            _save_encoded(f3_ast.apply_gcn(m, v, split_df, name, edge_type),
                           name, split_name, group)
        print(f"  ✅ {group} encoded")

    # F8/F9
    if 'F8' in needed and emb8 is not None:
        n_tr, n_va, n_te = len(tr), len(va), len(te)
        e8_tr = emb8[:n_tr]; e8_va = emb8[n_tr:n_tr+n_va]; e8_te = emb8[n_tr+n_va:]
        proj8 = f8_codebert.load_f8_proj()
        if n_tr: _save_encoded(f8_codebert.apply_f8(proj8, e8_tr), name, 'train', 'F8')
        if n_va: _save_encoded(f8_codebert.apply_f8(proj8, e8_va), name, 'val',   'F8')
        if n_te: _save_encoded(f8_codebert.apply_f8(proj8, e8_te), name, 'test',  'F8')
        print("  ✅ F8 encoded")

    if 'F9' in needed and emb9 is not None:
        n_tr, n_va, n_te = len(tr), len(va), len(te)
        e9_tr = emb9[:n_tr]; e9_va = emb9[n_tr:n_tr+n_va]; e9_te = emb9[n_tr+n_va:]
        proj9 = f9_graphcodebert.load_f9_proj()
        if n_tr: _save_encoded(f9_graphcodebert.apply_f9(proj9, e9_tr), name, 'train', 'F9')
        if n_va: _save_encoded(f9_graphcodebert.apply_f9(proj9, e9_va), name, 'val',   'F9')
        if n_te: _save_encoded(f9_graphcodebert.apply_f9(proj9, e9_te), name, 'test',  'F9')
        print("  ✅ F9 encoded")

    # F10
    if 'F10' in needed and X10 is not None:
        n_tr, n_va, n_te = len(tr), len(va), len(te)
        X10_tr = X10[:n_tr]; X10_va = X10[n_tr:n_tr+n_va]; X10_te = X10[n_tr+n_va:]
        proj10 = f10_security.load_f10_proj(X10.shape[1])
        if n_tr: _save_encoded(f10_security.apply_f10(proj10, X10_tr), name, 'train', 'F10')
        if n_va: _save_encoded(f10_security.apply_f10(proj10, X10_va), name, 'val',   'F10')
        if n_te: _save_encoded(f10_security.apply_f10(proj10, X10_te), name, 'test',  'F10')
        print("  ✅ F10 encoded")


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
    need_joern = any(g in needed for g in ['F3','F4','F5','F6','F7'])
    need_slice = 'F2' in needed
    if need_joern and len(list(jc.glob('*.json'))) < len(clean) * 0.5:
        try:
            print(f"⏳ Chạy Joern (skip_slice={not need_slice})...")
            JoernExtractor().extract_dataset(clean, name, skip_slice=not need_slice)
        except Exception as e:
            print(f"⚠️ Joern fail: {e}")

    # Fit encoder (chỉ Devign)
    if name == 'devign':
        fit_all_encoders(tr, va, name)

    # Transform subset
    transform_subset(name, clean, tr, va, te, needed)


def main(dataset):
    print_config(); set_seed(42)
    write_feature_config()

    if dataset == 'all':
        for t in ['devign', 'juliet', 'bigvul']:
            extract_dataset(t)
    else:
        extract_dataset(dataset)

    write_extraction_table('devign')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='all',
                    choices=['devign','juliet','bigvul','all'])
    main(ap.parse_args().dataset)