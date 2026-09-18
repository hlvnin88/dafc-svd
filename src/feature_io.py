
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from config import (DATA_DIR, F1_TFIDF_MAX_FEATURES, F1_NGRAM_RANGE,
                     F10_METRIC_NAMES, CODEBERT_NAME, GRAPHCODEBERT_NAME,
                     ENCODER_MAX_LEN, DEVICE)
from src.utils import ensure_dir


# ============================================================================
# F1 — F1_lexical_text.csv (TF-IDF 5,000-dim, CHƯA SVD)
# ============================================================================
def save_raw_f1(tr, va, te, name, vectorizer=None):
    """
    Fit TF-IDF (5,000-dim) trên train (lần đầu) hoặc dùng vectorizer đã có.
    Ghi 1 file CSV gộp cả 3 split.
    Trả về (path, vectorizer).
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    all_ids = (tr['sample_id'].tolist()
               + va['sample_id'].tolist()
               + te['sample_id'].tolist())
    all_lbl = (tr['label'].tolist()
               + va['label'].tolist()
               + te['label'].tolist())
    all_codes = (tr['source_code'].astype(str).tolist()
                 + va['source_code'].astype(str).tolist()
                 + te['source_code'].astype(str).tolist())

    if vectorizer is None:
        vectorizer = TfidfVectorizer(
            max_features=F1_TFIDF_MAX_FEATURES,
            ngram_range=F1_NGRAM_RANGE,
            token_pattern=r'\S+',
            lowercase=False,
        )
        X = vectorizer.fit_transform(all_codes)
    else:
        X = vectorizer.transform(all_codes)

    dense = X.toarray().astype(np.float32)

    out_dir = ensure_dir(DATA_DIR / name / 'features' / 'F1')
    out_path = out_dir / 'F1_lexical_text.csv'

    cols = [f'tfidf_{i}' for i in range(dense.shape[1])]
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('sample_id,label,' + ','.join(cols) + '\n')
        for i in range(dense.shape[0]):
            vals = ','.join(f'{v:.5g}' for v in dense[i])
            f.write(f'{all_ids[i]},{all_lbl[i]},{vals}\n')

    print(f"  💾 {out_path.relative_to(DATA_DIR.parents[0])}  "
          f"({dense.shape[0]} × {dense.shape[1]})")
    return str(out_path), vectorizer


# ============================================================================
# F2 — F2_sequence_slice.jsonl
# ============================================================================
def save_raw_f2(tr, va, te, name):
    """Đọc slice từ Joern cache, ghi 1 dòng JSONL / sample."""
    cache_dir = DATA_DIR / name / 'joern_raw'
    out_dir = ensure_dir(DATA_DIR / name / 'features' / 'F2')
    out_path = out_dir / 'F2_sequence_slice.jsonl'

    total = 0
    with open(out_path, 'w', encoding='utf-8') as f:
        for df in [tr, va, te]:
            for _, row in df.iterrows():
                sid = str(row['sample_id'])
                cj = cache_dir / f'{sid}.json'
                slices = []
                if cj.exists():
                    try:
                        with open(cj) as jf:
                            d = json.load(jf)
                        for s in (d.get('slices') or []):
                            if isinstance(s, dict):
                                slices.append(s)
                    except Exception:
                        pass
                rec = {
                    'sample_id': sid,
                    'label': int(row['label']),
                    'num_slices': len(slices),
                    'slices': slices,
                }
                f.write(json.dumps(rec) + '\n')
                total += 1

    print(f"  💾 {out_path.relative_to(DATA_DIR.parents[0])}  ({total} samples)")
    return str(out_path)


# ============================================================================
# F3–F7 — AST / CFG / DFG / CALL / CPG
# ============================================================================
_EDGE_TYPE_MAP = {
    'AST':  ('F3', 'F3_ast.jsonl'),
    'CFG':  ('F4', 'F4_cfg.pt'),
    'DDG':  ('F5', 'F5_dataflow_dependency.pt'),
    'CALL': ('F6', 'F6_call_interprocedural.pt'),
    'ALL':  ('F7', 'F7_composite_graph.pt'),
}


def save_raw_f3(clean, name, edge_type='AST'):
    """
    Đọc Joern cache → ghi:
        - AST → JSONL (F3_ast.jsonl)
        - CFG/DDG/CALL/ALL → torch .pt (F4..F7)
    """
    group, fname = _EDGE_TYPE_MAP[edge_type]
    cache_dir = DATA_DIR / name / 'joern_raw'
    out_dir = ensure_dir(DATA_DIR / name / 'features' / group)
    out_path = out_dir / fname

    if edge_type == 'AST':
        # JSONL
        total = 0
        with open(out_path, 'w', encoding='utf-8') as f:
            for _, row in clean.iterrows():
                sid = str(row['sample_id'])
                cj = cache_dir / f'{sid}.json'
                nodes, edges = [], []
                if cj.exists():
                    try:
                        with open(cj) as jf:
                            d = json.load(jf)
                        if d.get('nodes') is not None:
                            nodes = d['nodes'].get('label', [])
                        if d.get('rels') is not None:
                            r = d['rels']
                            srcs = r.get('src', [])
                            dsts = r.get('dst', [])
                            types = r.get('type', [])
                            for s, t, ty in zip(srcs, dsts, types):
                                if ty == 'AST':
                                    edges.append({'src': int(s), 'dst': int(t)})
                    except Exception:
                        pass
                rec = {
                    'sample_id': sid,
                    'label': int(row['label']),
                    'num_nodes': len(nodes),
                    'num_edges': len(edges),
                    'nodes': [str(n) for n in nodes],
                    'edges': edges,
                }
                f.write(json.dumps(rec) + '\n')
                total += 1
        print(f"  💾 {out_path.relative_to(DATA_DIR.parents[0])}  ({total} samples)")
    else:
        # .pt
        data = {}
        for _, row in clean.iterrows():
            sid = str(row['sample_id'])
            cj = cache_dir / f'{sid}.json'
            nodes, edges = [], []
            if cj.exists():
                try:
                    with open(cj) as jf:
                        d = json.load(jf)
                    if d.get('nodes') is not None:
                        nodes = d['nodes'].get('label', [])
                    if d.get('rels') is not None:
                        r = d['rels']
                        srcs = r.get('src', [])
                        dsts = r.get('dst', [])
                        types = r.get('type', [])
                        for s, t, ty in zip(srcs, dsts, types):
                            if edge_type == 'ALL' or ty == edge_type:
                                edges.append([int(s), int(t)])
                except Exception:
                    pass
            data[sid] = {
                'label': int(row['label']),
                'num_nodes': len(nodes),
                'nodes': [str(n) for n in nodes],
                'edges': torch.tensor(edges, dtype=torch.long)
                          if edges else torch.zeros((0, 2), dtype=torch.long),
            }
        torch.save(data, out_path)
        print(f"  💾 {out_path.relative_to(DATA_DIR.parents[0])}  ({len(data)} samples)")

    return str(out_path)


# ============================================================================
# F8 — F8_deep_semantic.npy (CodeBERT 768-dim)
# ============================================================================
def save_raw_f8(tr_codes, va_codes, te_codes, name):
    from transformers import AutoTokenizer, AutoModel

    print("  ⏳ CodeBERT encoding...")
    tok = AutoTokenizer.from_pretrained(CODEBERT_NAME)
    mdl = AutoModel.from_pretrained(CODEBERT_NAME).eval().to(DEVICE)

    out = []
    with torch.no_grad():
        for codes in [tr_codes, va_codes, te_codes]:
            for i in range(0, len(codes), 16):
                enc = tok(codes[i:i+16], return_tensors='pt',
                          truncation=True, max_length=ENCODER_MAX_LEN,
                          padding=True)
                enc = {k: v.to(DEVICE) for k, v in enc.items()}
                h = mdl(**enc).last_hidden_state.mean(dim=1)
                out.append(h.cpu().numpy().astype(np.float32))

    Z = np.concatenate(out, 0)
    out_dir = ensure_dir(DATA_DIR / name / 'features' / 'F8')
    out_path = out_dir / 'F8_deep_semantic.npy'
    np.save(out_path, Z)
    print(f"  💾 {out_path.relative_to(DATA_DIR.parents[0])}  shape={Z.shape}")
    return str(out_path), Z


# ============================================================================
# F9 — F9_deep_structural.npy (GraphCodeBERT 768-dim)
# ============================================================================
def save_raw_f9(tr_codes, va_codes, te_codes, name):
    from transformers import AutoTokenizer, AutoModel

    print("  ⏳ GraphCodeBERT encoding...")
    tok = AutoTokenizer.from_pretrained(GRAPHCODEBERT_NAME)
    mdl = AutoModel.from_pretrained(GRAPHCODEBERT_NAME).eval().to(DEVICE)

    out = []
    with torch.no_grad():
        for codes in [tr_codes, va_codes, te_codes]:
            for i in range(0, len(codes), 16):
                enc = tok(codes[i:i+16], return_tensors='pt',
                          truncation=True, max_length=ENCODER_MAX_LEN,
                          padding=True)
                enc = {k: v.to(DEVICE) for k, v in enc.items()}
                h = mdl(**enc).last_hidden_state.mean(dim=1)
                out.append(h.cpu().numpy().astype(np.float32))

    Z = np.concatenate(out, 0)
    out_dir = ensure_dir(DATA_DIR / name / 'features' / 'F9')
    out_path = out_dir / 'F9_deep_structural.npy'
    np.save(out_path, Z)
    print(f"  💾 {out_path.relative_to(DATA_DIR.parents[0])}  shape={Z.shape}")
    return str(out_path), Z


# ============================================================================
# F10 — F10_security_metrics.csv (32 metrics)
# ============================================================================
_DANGEROUS = ['strcpy', 'strcat', 'sprintf', 'gets',
              'malloc', 'free', 'memcpy']


def _f10_metrics(code):
    """Trả về dict 32 metrics (đúng thứ tự F10_METRIC_NAMES)."""
    if not isinstance(code, str):
        return {m: 0 for m in F10_METRIC_NAMES}

    lines = code.split('\n')

    def cnt(pat):
        return len(re.findall(pat, code))

    vals = {
        'num_lines': len(lines),
        'num_functions': len(re.findall(r'\b\w+\s+\w+\s*\([^;]*\)\s*\{', code)),
        'num_params': cnt(r','),
        'cyclomatic_complexity': 1 + cnt(r'\b(if|for|while|case|&&|\|\|)\b'),
        'nesting_depth': max([l.count('{') for l in lines] + [0]),
        'num_branches': cnt(r'\b(if|else|switch|case)\b'),
        'num_loops': cnt(r'\b(for|while|do)\b'),
        'num_returns': cnt(r'\breturn\b'),
        'num_dangerous_apis': sum(cnt(rf'\b{d}\b') for d in _DANGEROUS),
        'num_strcpy': cnt(r'\bstrcpy\b'),
        'num_strcat': cnt(r'\bstrcat\b'),
        'num_sprintf': cnt(r'\bsprintf\b'),
        'num_gets': cnt(r'\bgets\b'),
        'num_malloc': cnt(r'\bmalloc\b'),
        'num_free': cnt(r'\bfree\b'),
        'num_memcpy': cnt(r'\bmemcpy\b'),
        'has_pointer': int('*' in code),
        'num_pointer_derefs': cnt(r'\*'),
        'has_array': int('[' in code),
        'num_array_index': cnt(r'\['),
        'has_ampersand': int('&' in code),
        'num_bit_ops': cnt(r'(<<|>>|&|\||\^|~)'),
        'has_union': int(bool(re.search(r'\bunion\b', code))),
        'has_void_ptr': int('void' in code and '*' in code),
        'has_bounds_check': int(bool(re.search(
            r'(if|while)\s*\([^)]*<[^)]*\)', code))),
        'has_null_check': int('NULL' in code or 'nullptr' in code),
        'has_size_check': int(bool(re.search(r'\bsizeof\b', code))),
        'num_arithmetic_ops': cnt(r'[\+\-\*/%]'),
        'num_assignments': cnt(r'='),
        'num_comparisons': cnt(r'(==|!=|<=|>=|<|>)'),
        'num_casts': cnt(r'\(\s*(int|char|float|double|void)\s*\*?\s*\)'),
        'num_goto': cnt(r'\bgoto\b'),
    }
    return {m: vals.get(m, 0) for m in F10_METRIC_NAMES}


def save_raw_f10(tr, va, te, name):
    """Ghi F10_security_metrics.csv (32 cột). Trả về (path, X_metrics)."""
    rows = []
    for df in [tr, va, te]:
        for _, r in df.iterrows():
            row = {'sample_id': r['sample_id'], 'label': int(r['label'])}
            row.update(_f10_metrics(str(r['source_code'])))
            rows.append(row)

    df = pd.DataFrame(rows)
    out_dir = ensure_dir(DATA_DIR / name / 'features' / 'F10')
    out_path = out_dir / 'F10_security_metrics.csv'
    df.to_csv(out_path, index=False)
    print(f"  💾 {out_path.relative_to(DATA_DIR.parents[0])}  shape={df.shape}")

    X = df[F10_METRIC_NAMES].values.astype(np.float32)
    return str(out_path), X


# ============================================================================
# HELPER — Tên file cho Bảng 4
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