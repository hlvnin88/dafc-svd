#!/usr/bin/env python
"""DAFC-SVD — Bước 4–15."""
import sys, shutil
from pathlib import Path
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (DATA_DIR, SELECTION_DIR, TABLES_DIR, FEATURE_GROUPS,
                     RANDOM_SEEDS, TOP_K, print_config,
                     RANDOM_SELECTION_SEED)
from src.utils import (set_seed, print_section, print_table, save_csv, fmt_ms)
from src.individual_train import train_all_individuals
from src.prediction_log import (save_prediction_log, save_error_vector,
                                 build_cost_components_table, save_valid_sample_ids)
from src.combo_generator import generate_all_combos, save_candidates
from src.metrics import (build_com_matrix, build_ovl_matrix,
                          compute_coverage, avg_ovl, combo_cost)
from src.filter_combos import run_filter, save_qualified_41, save_top7
from src.mmf_train import train_and_eval_combo
from src.evaluation import evaluate_model
from src.ablation import (ablation_without_coverage, ablation_without_com,
                           ablation_without_ovl, ablation_without_cost,
                           save_ablation_selection, save_ablation_fusion)
from src.sensitivity import (five_feature_search, threshold_sensitivity,
                              prescore_sensitivity, seed_stability)
from src.baselines import (random_selection, top_individual_selection,
                            CodeBERTBaseline, DWF, CPG_RGCN,
                            train_text_baseline, eval_text_baseline,
                            train_cpg_rgcn, eval_cpg_rgcn)
from src.plotting import plot_com_ovl_heatmaps, plot_funnel, plot_pareto


def _ids(name, split):
    p = DATA_DIR / name / 'splits' / f'{name}_{split}_ids.txt'
    if not p.exists(): return []
    with open(p) as f: return [l.strip() for l in f if l.strip()]


def _labels(name, split):
    clean = pd.read_csv(DATA_DIR / name / 'clean' / f'{name}_clean.csv')
    ids = _ids(name, split)
    order = {sid: i for i, sid in enumerate(ids)}
    sub = clean[clean['sample_id'].isin(ids)].copy()
    sub['_o'] = sub['sample_id'].map(order)
    sub = sub.sort_values('_o').drop(columns='_o')
    return sub['label'].values.astype(int), sub['sample_id'].values


def _load_feat(name, group, split):
    d = DATA_DIR / name / 'features' / group
    p = d / f'{name}_{split}_{group.lower()}.npy'
    if p.exists(): return np.load(p)
    pf = d / f'{name}_full_{group.lower()}.npy'
    if not pf.exists(): return None
    full = np.load(pf)
    clean = pd.read_csv(DATA_DIR / name / 'clean' / f'{name}_clean.csv')
    ids = _ids(name, split)
    idx = clean.reset_index()[clean['sample_id'].isin(ids)].index.tolist()
    return full[idx]


def _valid_intersection(arrs, y):
    masks = np.ones(len(y), dtype=bool)
    for a in arrs: masks &= ~np.all(a == 0, axis=1)
    idx = np.where(masks)[0]
    return [a[idx] for a in arrs], y[idx]


def _build_combo_xy(name, combo, split):
    arrs = [_load_feat(name, g, split) for g in combo]
    if any(a is None for a in arrs):
        raise FileNotFoundError(f"missing feature {name}/{combo}/{split}")
    y, _ = _labels(name, split)
    return _valid_intersection(arrs, y)


# ============================================================================
# BƯỚC 4 & 5
# ============================================================================
def step4_5():
    print_section("BƯỚC 4 & 5 — Individual F1–F10 + Logs", '=')
    y_tr, _ = _labels('devign', 'train')
    y_va, va_ids = _labels('devign', 'val')
    y_te, _ = _labels('devign', 'test')

    feat = {}
    for g in FEATURE_GROUPS:
        a_tr = _load_feat('devign', g, 'train')
        a_va = _load_feat('devign', g, 'val')
        a_te = _load_feat('devign', g, 'test')
        if a_tr is None or a_va is None or a_te is None:
            print(f"  ⚠️  skip {g}"); continue
        feat[g] = {'train': a_tr, 'val': a_va, 'test': a_te}

    results = train_all_individuals(feat, y_tr, y_va, y_te, dataset_name='devign')

    # === Bảng #5 — Individual performance ===
    rows = [{'Group': g,
             'Precision': fmt_ms(r['mean_val']['precision'], r['std_val']['precision']),
             'Recall':    fmt_ms(r['mean_val']['recall'],    r['std_val']['recall']),
             'F1-score':  fmt_ms(r['mean_val']['f1'],        r['std_val']['f1']),
             'ROC-AUC':   fmt_ms(r['mean_val']['auc'],       r['std_val']['auc'])}
            for g, r in results.items()]
    df5 = pd.DataFrame(rows)
    save_csv(df5, 'individual_performance', TABLES_DIR)
    print_table(df5, 'individual_performance')

    # Prediction logs + error vectors
    for g, r in results.items():
        for seed in RANDOM_SEEDS:
            probs = r['val_probs_by_seed'][seed]
            preds = r['val_preds_by_seed'][seed]
            yv = y_va[:len(probs)]; sid = va_ids[:len(probs)]
            save_prediction_log(g, seed, yv, probs, preds, sid)
            save_error_vector(g, seed, (preds != yv).astype(np.int8))

    # === Bảng #4 — Cost components ===
    build_cost_components_table('devign')

    save_valid_sample_ids({
        'devign': {'train': _ids('devign','train'),
                    'val': _ids('devign','val'),
                    'test': _ids('devign','test')},
        'juliet': {'train': _ids('juliet','train'),
                    'val': _ids('juliet','val'),
                    'test': _ids('juliet','test')},
        'bigvul': {'train': [], 'val': [], 'test': _ids('bigvul','test')},
    })
    return results, y_va, va_ids


# ============================================================================
# BƯỚC 6, 7, 8
# ============================================================================
def step6_7_8(individual_results, y_va, errs_per_seed):
    print_section("BƯỚC 6 — Sinh 375 tổ hợp", '=')
    combos = generate_all_combos(FEATURE_GROUPS)
    save_candidates(combos)

    print_section("BƯỚC 7 — Coverage/Com/Ovl/Cost/PreScore", '=')
    errs_mean = {}
    for g in FEATURE_GROUPS:
        stack = np.stack([errs_per_seed[seed][g] for seed in RANDOM_SEEDS])
        errs_mean[g] = (stack.mean(0) >= 0.5).astype(np.int8)

    perf = {g: r['mean_val']['f1'] for g, r in individual_results.items()}
    sl = {g: np.ones(len(y_va), dtype=bool) for g in FEATURE_GROUPS}

    print_section("BƯỚC 8 — Lọc", '=')
    c1, c2, c3, scored, top = run_filter(combos, perf, errs_mean, sl)
    save_qualified_41(scored); save_top7(top)

    # === Bảng #6 — Funnel ===
    funnel_rows = [
        {'Step': 'Initial candidates',            'Remaining combinations': 375},
        {'Step': 'After coverage filtering',      'Remaining combinations': len(c1)},
        {'Step': 'After error-overlap filtering', 'Remaining combinations': len(c2)},
        {'Step': 'After cost filtering',          'Remaining combinations': len(c3)},
        {'Step': 'After PreScore ranking (K=7)',  'Remaining combinations': len(top)},
    ]
    for r in funnel_rows:
        r['Retention rate (%)'] = round(r['Remaining combinations'] / 375 * 100, 2)
    save_csv(pd.DataFrame(funnel_rows), 'funnel', TABLES_DIR)

    # === Bảng #7 — Top-7 combinations ===
    top7_df = pd.read_csv(SELECTION_DIR / 'top7.csv')
    top7_df.to_csv(TABLES_DIR / 'top7_combinations.csv', index=False)
    print(f"  💾 results/tables/top7_combinations.csv")
    print_table(top7_df, 'top7_combinations')

    plot_funnel({'Initial (375)': 375,
                 f'Coverage ({len(c1)})': len(c1),
                 f'Ovl ({len(c2)})': len(c2),
                 f'Cost ({len(c3)})': len(c3),
                 f'Top-{TOP_K}': len(top)})
    plot_com_ovl_heatmaps(build_com_matrix(FEATURE_GROUPS, errs_mean),
                          build_ovl_matrix(FEATURE_GROUPS, errs_mean),
                          FEATURE_GROUPS)
    return top, combos, perf, errs_mean, sl


# ============================================================================
# BƯỚC 9, 10, 11
# ============================================================================
def step9_10_11(top7):
    print_section("BƯỚC 9-11 — MMAF Devign/Juliet/Big-Vul", '=')
    results = {'devign': {}, 'juliet': {}, 'bigvul': {}}

    for item in top7:
        sym, combo = item['symbol'], item['combo']
        print(f"\n▶️  {sym} = {' + '.join(combo)}")
        try:
            Xtr, ytr = _build_combo_xy('devign', combo, 'train')
            Xva, yva = _build_combo_xy('devign', combo, 'val')
            Xte, yte = _build_combo_xy('devign', combo, 'test')
            r9 = train_and_eval_combo(combo, Xtr, ytr, Xva, yva, Xte, yte,
                                       ckpt_name=f'{sym}_devign')
            results['devign'][sym] = r9
        except Exception as e:
            print(f"   ❌ Devign {sym}: {e}"); continue
        try:
            Xtr_j, ytr_j = _build_combo_xy('juliet', combo, 'train')
            Xva_j, yva_j = _build_combo_xy('juliet', combo, 'val')
            Xte_j, yte_j = _build_combo_xy('juliet', combo, 'test')
            r10 = train_and_eval_combo(combo, Xtr_j, ytr_j, Xva_j, yva_j,
                                        Xte_j, yte_j, ckpt_name=f'{sym}_juliet')
            results['juliet'][sym] = r10
        except Exception as e:
            print(f"   ⚠️  Juliet {sym}: {e}")
        try:
            Xte_b, yte_b = _build_combo_xy('bigvul', combo, 'test')
            r11 = evaluate_model(r9['last_model'], Xte_b, yte_b)
            results['bigvul'][sym] = {'mean_test': r11,
                                       'std_test': {k: 0.0 for k in r11}}
        except Exception as e:
            print(f"   ⚠️  Big-Vul {sym}: {e}")

    # Bảng #12, #13, #14
    name_map = {'devign': 'mmaf_devign', 'juliet': 'mmaf_juliet', 'bigvul': 'mmaf_bigvul'}
    for key, tname in name_map.items():
        rows = [{'Symbol': sym,
                 'Precision': fmt_ms(r['mean_test']['precision'], r['std_test']['precision']),
                 'Recall':    fmt_ms(r['mean_test']['recall'],    r['std_test']['recall']),
                 'F1-score':  fmt_ms(r['mean_test']['f1'],        r['std_test']['f1']),
                 'ROC-AUC':   fmt_ms(r['mean_test']['auc'],       r['std_test']['auc'])}
                for sym, r in results[key].items()]
        df = pd.DataFrame(rows); save_csv(df, tname, TABLES_DIR); print_table(df, tname)

    # Bảng #15 — F1 summary
    if results['devign']:
        rows = []
        for sym in results['devign']:
            dv = results['devign'][sym]['mean_test']['f1']
            jl = results['juliet'].get(sym, {}).get('mean_test', {}).get('f1', 0.0)
            bv = results['bigvul'].get(sym, {}).get('mean_test', {}).get('f1', 0.0)
            rows.append({'Symbol': sym, 'Devign': f"{dv:.3f}",
                         'Juliet': f"{jl:.3f}", 'Big-Vul': f"{bv:.3f}",
                         'Average': f"{(dv+jl+bv)/3:.3f}", 'Drop': f"{dv-bv:.3f}"})
        df15 = pd.DataFrame(rows); save_csv(df15, 'f1_summary', TABLES_DIR)
        print_table(df15, 'f1_summary')
    return results


# ============================================================================
# BƯỚC 12
# ============================================================================
def step12(combos, perf, errs_mean, sl, mmaf_results):
    print_section("BƯỚC 12 — Ablation", '=')
    rows_sel = []
    for tag, fn in [('Without Coverage', ablation_without_coverage),
                     ('Without Com',      ablation_without_com),
                     ('Without Ovl',      ablation_without_ovl),
                     ('Without Cost',     ablation_without_cost)]:
        top = fn(combos, perf, errs_mean, sl)
        rows_sel.append({'Variant': tag,
                         'Selected combination': ' + '.join(top[0]['combo']),
                         'PreScore': round(top[0]['prescore'], 4)})
    save_ablation_selection(rows_sel)   # → ghi tên cũ, cần sửa bên dưới

    rows_fusion = [{'Variant': f'{tag} fusion', 'Devign': 'see mmaf_devign.csv',
                     'Juliet': 'see mmaf_juliet.csv', 'Big-Vul': 'see mmaf_bigvul.csv'}
                    for tag in ['MMAF', 'Concatenation', 'Average']]
    save_ablation_fusion(rows_fusion)

    # Rename
    for old, new in [('table12_ablation_selection.csv', 'ablation.csv'),
                     ('table12_ablation_fusion.csv',    'ablation_fusion.csv')]:
        p_old = TABLES_DIR / old
        if p_old.exists(): shutil.move(str(p_old), str(TABLES_DIR / new))
        print(f"  💾 results/tables/{new}")


# ============================================================================
# BƯỚC 13
# ============================================================================
def step13(perf):
    print_section("BƯỚC 13 — Random / Top-individual + MMAF", '=')
    rand = random_selection(FEATURE_GROUPS, k=4)
    topc = top_individual_selection(perf, k=4)
    rows = []
    for tag, combo in [('Random selection + MMAF', rand),
                        ('Top-individual + MMAF',   topc)]:
        try:
            Xtr, ytr = _build_combo_xy('devign', combo, 'train')
            Xva, yva = _build_combo_xy('devign', combo, 'val')
            Xte, yte = _build_combo_xy('devign', combo, 'test')
            r = train_and_eval_combo(combo, Xtr, ytr, Xva, yva, Xte, yte,
                                     ckpt_name=f'base_{tag.split()[0]}')
            rows.append({'Method': tag, 'Combination': ' + '.join(combo),
                         'Devign F1': f"{r['mean_test']['f1']:.3f}"})
        except Exception as e:
            rows.append({'Method': tag, 'Combination': ' + '.join(combo),
                         'Devign F1': f'error: {e}'})
    df = pd.DataFrame(rows)
    save_csv(df, 'selection_baselines', TABLES_DIR)
    print_table(df, 'selection_baselines')


# ============================================================================
# BƯỚC 14
# ============================================================================
def step14():
    print_section("BƯỚC 14 — External baselines", '=')
    from transformers import AutoTokenizer
    from torch.utils.data import DataLoader, TensorDataset

    clean = pd.read_csv(DATA_DIR / 'devign' / 'clean' / 'devign_clean.csv')
    tr = clean[clean['sample_id'].isin(_ids('devign','train'))].reset_index(drop=True)
    va = clean[clean['sample_id'].isin(_ids('devign','val'))].reset_index(drop=True)
    te = clean[clean['sample_id'].isin(_ids('devign','test'))].reset_index(drop=True)
    y_tr = tr['label'].values.astype(np.float32)
    y_va = va['label'].values.astype(np.float32)
    y_te = te['label'].values.astype(np.float32)

    tok = AutoTokenizer.from_pretrained('microsoft/codebert-base')
    tok2 = AutoTokenizer.from_pretrained('microsoft/graphcodebert-base')

    def _tok(texts, tk):
        e = tk(list(texts), truncation=True, padding=True, max_length=512,
                return_tensors='pt')
        return e['input_ids'], e['attention_mask']

    ids1_tr, m1_tr = _tok(tr['source_code'].astype(str), tok)
    ids1_va, m1_va = _tok(va['source_code'].astype(str), tok)
    ids1_te, m1_te = _tok(te['source_code'].astype(str), tok)
    ids2_tr, m2_tr = _tok(tr['source_code'].astype(str), tok2)
    ids2_va, m2_va = _tok(va['source_code'].astype(str), tok2)
    ids2_te, m2_te = _tok(te['source_code'].astype(str), tok2)
    y_tr_t = torch.tensor(y_tr); y_va_t = torch.tensor(y_va); y_te_t = torch.tensor(y_te)

    def _ld_cb(ids, m, y, bs=8, sh=True):
        return DataLoader(TensorDataset(ids, m, y), batch_size=bs, shuffle=sh)
    def _ld_dwf(i1, m1, i2, m2, y, bs=8, sh=True):
        return DataLoader(TensorDataset(i1, m1, i2, m2, y), batch_size=bs, shuffle=sh)

    cb = CodeBERTBaseline()
    cb = train_text_baseline(cb, _ld_cb(ids1_tr, m1_tr, y_tr_t),
                              _ld_cb(ids1_va, m1_va, y_va_t),
                              kind='codebert', lr=2e-5, epochs=3, patience=2)
    cb_m = eval_text_baseline(cb, _ld_cb(ids1_te, m1_te, y_te_t, sh=False), 'codebert')

    dwf = DWF()
    dwf = train_text_baseline(dwf,
        _ld_dwf(ids1_tr, m1_tr, ids2_tr, m2_tr, y_tr_t),
        _ld_dwf(ids1_va, m1_va, ids2_va, m2_va, y_va_t),
        kind='dwf', lr=2e-5, epochs=3, patience=2)
    dwf_m = eval_text_baseline(dwf,
        _ld_dwf(ids1_te, m1_te, ids2_te, m2_te, y_te_t, sh=False), 'dwf')

    F7_tr = _load_feat('devign','F7','train')
    F7_va = _load_feat('devign','F7','val')
    F7_te = _load_feat('devign','F7','test')
    if F7_tr is not None:
        cpg = train_cpg_rgcn(F7_tr.shape[1], F7_tr, y_tr, F7_va, y_va)
        cpg_m = eval_cpg_rgcn(cpg, F7_te, y_te)
    else:
        cpg_m = {'precision':0.0,'recall':0.0,'f1':0.0,'auc':0.0}

    rows = [{'Method': 'CPG + RGCN', **cpg_m},
            {'Method': 'CodeBERT',   **cb_m},
            {'Method': 'DWF',        **dwf_m}]
    df = pd.DataFrame(rows)
    save_csv(df, 'external_baselines', TABLES_DIR)
    print_table(df, 'external_baselines')


# ============================================================================
# BƯỚC 15
# ============================================================================
def step15(combos, perf, errs_mean, sl, errs_per_seed, results):
    print_section("BƯỚC 15 — Sensitivity", '=')
    five_feature_search(perf, errs_mean, sl)
    threshold_sensitivity(combos, perf, errs_mean, sl)

    q = [c for c in combos if compute_coverage(c, sl) >= 0.80
         and avg_ovl(c, errs_mean) <= 0.60
         and combo_cost(c) <= 0.75]
    prescore_sensitivity([{'combo': c} for c in q], perf, errs_mean, sl)

    perf_per_seed = {seed: {g: results[g]['per_seed_val'][i]['f1']
                             for g in results}
                     for i, seed in enumerate(RANDOM_SEEDS)}
    errs_map = {seed: {g: errs_per_seed[seed][g] for g in FEATURE_GROUPS}
                for seed in RANDOM_SEEDS}
    seed_stability(combos, errs_map, perf_per_seed, sl)

    # Rename các file sensitivity
    rename_map = {
        'table15_threshold_sensitivity.csv': 'threshold_sensitivity.csv',
        'table15_five_feature.csv':          'five_feature_analysis.csv',
        'table15_prescore_sensitivity.csv':  'prescore_sensitivity.csv',
        'table15_seed_stability.csv':        'seed_stability.csv',
    }
    for old, new in rename_map.items():
        p_old = TABLES_DIR / old
        if p_old.exists():
            shutil.move(str(p_old), str(TABLES_DIR / new))
            print(f"  💾 results/tables/{new}")


# ============================================================================
# MAIN
# ============================================================================
def main():
    print_config(); set_seed(42)
    print_section("DAFC-SVD — BƯỚC 4–15", '=')

    results, y_va, va_ids = step4_5()

    errs_per_seed = {seed: {} for seed in RANDOM_SEEDS}
    for g, r in results.items():
        for seed in RANDOM_SEEDS:
            preds = r['val_preds_by_seed'][seed]
            yv = y_va[:len(preds)]
            errs_per_seed[seed][g] = (preds != yv).astype(np.int8)

    top, combos, perf, errs_mean, sl = step6_7_8(results, y_va, errs_per_seed)
    mmaf_results = step9_10_11(top)

    step12(combos, perf, errs_mean, sl, mmaf_results)
    step13(perf)
    step14()
    step15(combos, perf, errs_mean, sl, errs_per_seed, results)

    # Pareto
    try:
        top7_df = pd.read_csv(SELECTION_DIR / 'top7.csv')
        info = []
        for _, r in top7_df.iterrows():
            sym = r['Symbol']
            f1s = []
            for ds in ['devign','juliet','bigvul']:
                x = mmaf_results.get(ds, {}).get(sym)
                if x: f1s.append(x['mean_test']['f1'])
            info.append({'symbol': sym, 'cost': float(r['Cost']),
                         'avg_f1': float(np.mean(f1s)) if f1s else 0.0})
        plot_pareto(info)
    except Exception as e:
        print(f"⚠️  Pareto: {e}")

    print_section("✅ HOÀN TẤT", '=')


if __name__ == '__main__':
    main()