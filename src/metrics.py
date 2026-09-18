"""Bước 7: Coverage / BasePerf / Com / Ovl / Cost / PreScore."""
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import combinations
from config import ALPHA, BETA, GAMMA, ETA, COST_LOGS_DIR


_COST_CACHE = None

def _load_cost_table():
    global _COST_CACHE
    if _COST_CACHE is not None: return _COST_CACHE
    p = Path(COST_LOGS_DIR) / 'cost_log.csv'
    if not p.exists():
        raise RuntimeError(f"Chưa có {p}. Chạy Bước 3 + Bước 5 trước.")
    df = pd.read_csv(p)
    _COST_CACHE = dict(zip(df['group'], df['cost']))
    return _COST_CACHE


def group_cost(g): return float(_load_cost_table().get(g, 0.0))
def combo_cost(combo): return float(np.mean([group_cost(g) for g in combo]))


def compute_coverage(combo, success_logs):
    masks = [success_logs[g].astype(bool) for g in combo if g in success_logs]
    if not masks: return 0.0
    valid = np.all(np.stack(masks, 0), 0)
    return float(valid.mean()) if len(valid) else 0.0


def com_pair(ea, eb):
    a = (ea == 1).sum()
    if a == 0: return 0.0
    return float(((ea == 1) & (eb == 0)).sum() / a)

def com_pair_sym(ea, eb): return 0.5*(com_pair(ea, eb) + com_pair(eb, ea))

def avg_com(combo, errs):
    if len(combo) < 2: return 0.0
    return float(np.mean([com_pair_sym(errs[a], errs[b]) for a,b in combinations(combo, 2)]))


def ovl_pair(ea, eb):
    both = ((ea==1)&(eb==1)).sum(); either = ((ea==1)|(eb==1)).sum()
    return float(both/either) if either > 0 else 0.0

def avg_ovl(combo, errs):
    if len(combo) < 2: return 0.0
    return float(np.mean([ovl_pair(errs[a], errs[b]) for a,b in combinations(combo, 2)]))


def base_perf(combo, perf_dict):
    vals = [perf_dict[g] for g in combo if g in perf_dict]
    return float(np.mean(vals)) if vals else 0.0


def prescore(combo, perf_dict, errs, success_logs,
             alpha=ALPHA, beta=BETA, gamma=GAMMA, eta=ETA):
    bp = base_perf(combo, perf_dict); cm = avg_com(combo, errs)
    ov = avg_ovl(combo, errs); ct = combo_cost(combo)
    cov = compute_coverage(combo, success_logs)
    score = alpha*bp + beta*cm - gamma*ov - eta*ct
    return {'prescore': score, 'coverage': cov, 'com': cm,
            'ovl': ov, 'cost': ct, 'baseperf': bp}


def build_com_matrix(groups, errs):
    n = len(groups); M = np.zeros((n,n))
    for i,a in enumerate(groups):
        for j,b in enumerate(groups):
            M[i,j] = np.nan if i==j else com_pair_sym(errs[a], errs[b])
    return M

def build_ovl_matrix(groups, errs):
    n = len(groups); M = np.zeros((n,n))
    for i,a in enumerate(groups):
        for j,b in enumerate(groups):
            M[i,j] = np.nan if i==j else ovl_pair(errs[a], errs[b])
    return M