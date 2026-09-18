"""Bước 15: Sensitivity."""
import itertools
import numpy as np
import pandas as pd
from config import (FEATURE_GROUPS, THRESHOLD_CONFIGS, PRESCORE_CONFIGS,
                     THETA_COV, THETA_OVL, THETA_COST, ALPHA, BETA, GAMMA, ETA,
                     RANDOM_SEEDS, TABLES_DIR)
from src.metrics import compute_coverage, avg_ovl, combo_cost, prescore
from src.utils import save_csv


def five_feature_search(perf, errs, sl):
    combos = list(itertools.combinations(FEATURE_GROUPS, 5))
    c1 = [c for c in combos if compute_coverage(c, sl) >= THETA_COV]
    c2 = [c for c in c1 if avg_ovl(c, errs) <= THETA_OVL]
    c3 = [c for c in c2 if combo_cost(c) <= THETA_COST]
    scored = []
    for c in c3:
        s = prescore(c, perf, errs, sl)
        scored.append({'combo': c, **s})
    scored.sort(key=lambda x: x['prescore'], reverse=True)
    rows = [{'Rank': i, 'Feature combination': ' + '.join(s['combo']),
             'Coverage': round(s['coverage'],4), 'Com': round(s['com'],4),
             'Ovl': round(s['ovl'],4), 'Cost': round(s['cost'],4),
             'PreScore': round(s['prescore'],4)}
            for i, s in enumerate(scored, 1)]
    df = pd.DataFrame(rows)
    save_csv(df, 'table15_five_feature', TABLES_DIR)
    print(f"   5-feature: {len(combos)} → {len(c1)} → {len(c2)} → {len(c3)}")
    return df


def threshold_sensitivity(combos, perf, errs, sl):
    rows = []
    for name, th in THRESHOLD_CONFIGS.items():
        c1 = [c for c in combos if compute_coverage(c, sl) >= th['cov']]
        c2 = [c for c in c1 if avg_ovl(c, errs) <= th['ovl']]
        c3 = [c for c in c2 if combo_cost(c) <= th['cost']]
        rows.append({'Configuration': name,
                     'θ_cov': th['cov'], 'θ_ovl': th['ovl'], 'θ_cost': th['cost'],
                     'After_Coverage': len(c1), 'After_Ovl': len(c2),
                     'After_Cost': len(c3)})
    df = pd.DataFrame(rows)
    save_csv(df, 'table15_threshold_sensitivity', TABLES_DIR)
    return df


def prescore_sensitivity(qualified, perf, errs, sl):
    rows = []
    default_top7 = None
    for name, w in PRESCORE_CONFIGS.items():
        scored = []
        for item in qualified:
            c = item['combo']
            s = prescore(c, perf, errs, sl,
                         alpha=w['alpha'], beta=w['beta'],
                         gamma=w['gamma'], eta=w['eta'])
            scored.append({'combo': c, **s})
        scored.sort(key=lambda x: x['prescore'], reverse=True)
        top7 = [s['combo'] for s in scored[:7]]
        if name == 'Default': default_top7 = set(top7)
        rows.append({'Configuration': name, 'α': w['alpha'], 'β': w['beta'],
                     'γ': w['gamma'], 'η': w['eta'],
                     'Top-7': [' + '.join(c) for c in top7]})
    for r in rows:
        top7 = set(tuple(s.split(' + ')) for s in r['Top-7'])
        r['Overlap_with_Default'] = f"{len(top7 & default_top7) if default_top7 else 0}/7"
    df = pd.DataFrame(rows)
    save_csv(df, 'table15_prescore_sensitivity', TABLES_DIR)
    return df


def seed_stability(combos, per_seed_errs, per_seed_perf, sl):
    rows = []
    default_top7 = None
    for seed in RANDOM_SEEDS:
        errs = per_seed_errs[seed]; perf = per_seed_perf[seed]
        c1 = [c for c in combos if compute_coverage(c, sl) >= THETA_COV]
        c2 = [c for c in c1 if avg_ovl(c, errs) <= THETA_OVL]
        c3 = [c for c in c2 if combo_cost(c) <= THETA_COST]
        scored = []
        for c in c3:
            s = prescore(c, perf, errs, sl, alpha=ALPHA, beta=BETA, gamma=GAMMA, eta=ETA)
            scored.append({'combo': c, **s})
        scored.sort(key=lambda x: x['prescore'], reverse=True)
        top7 = set(s['combo'] for s in scored[:7])
        if default_top7 is None: default_top7 = top7
        rows.append({'Seed': seed, 'Top-7': [' + '.join(c) for c in top7],
                     'Overlap_with_seed42': f"{len(top7 & default_top7)}/7"})
    df = pd.DataFrame(rows)
    save_csv(df, 'table15_seed_stability', TABLES_DIR)
    return df