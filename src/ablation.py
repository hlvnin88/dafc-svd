"""Bước 12: Ablation."""
import pandas as pd
from config import THETA_OVL, THETA_COST, THETA_COV, BETA, GAMMA, ETA, TABLES_DIR
from src.metrics import compute_coverage, avg_ovl, combo_cost, prescore
from src.utils import save_csv


def _rank(combos, perf_dict, errs, success_logs, top_k=7, remove=None):
    scored = []
    for c in combos:
        s = prescore(c, perf_dict, errs, success_logs)
        if remove == 'com':  s['prescore'] -= BETA  * s['com']
        if remove == 'ovl':  s['prescore'] += GAMMA * s['ovl']
        if remove == 'cost': s['prescore'] += ETA   * s['cost']
        scored.append({'combo': c, **s})
    scored.sort(key=lambda x: x['prescore'], reverse=True)
    return scored[:top_k]


def _filter(combos, errs, sl, use_cov=True, use_ovl=True, use_cost=True):
    c = list(combos)
    if use_cov:  c = [x for x in c if compute_coverage(x, sl) >= THETA_COV]
    if use_ovl:  c = [x for x in c if avg_ovl(x, errs) <= THETA_OVL]
    if use_cost: c = [x for x in c if combo_cost(x) <= THETA_COST]
    return c


def ablation_without_coverage(combos, perf, errs, sl):
    return _rank(_filter(combos, errs, sl, use_cov=False), perf, errs, sl)

def ablation_without_com(combos, perf, errs, sl):
    return _rank(_filter(combos, errs, sl), perf, errs, sl, remove='com')

def ablation_without_ovl(combos, perf, errs, sl):
    return _rank(_filter(combos, errs, sl, use_ovl=False), perf, errs, sl, remove='ovl')

def ablation_without_cost(combos, perf, errs, sl):
    return _rank(_filter(combos, errs, sl, use_cost=False), perf, errs, sl, remove='cost')


def save_ablation_selection(rows):
    df = pd.DataFrame(rows)
    save_csv(df, 'table12_ablation_selection', TABLES_DIR)
    return df

def save_ablation_fusion(rows):
    df = pd.DataFrame(rows)
    save_csv(df, 'table12_ablation_fusion', TABLES_DIR)
    return df