"""Bước 8: Lọc."""
import pandas as pd
from config import THETA_COV, THETA_OVL, THETA_COST, TOP_K, SELECTION_DIR
from src.metrics import compute_coverage, avg_ovl, combo_cost, prescore
from src.utils import ensure_dir

def run_filter(combos, perf_dict, errs, success_logs,
               cov_th=THETA_COV, ovl_th=THETA_OVL, cost_th=THETA_COST, top_k=TOP_K):
    print(f"📊 Lọc {len(combos)} tổ hợp")
    c1 = [c for c in combos if compute_coverage(c, success_logs) >= cov_th]
    print(f"   Coverage≥{cov_th}: {len(c1)}")
    c2 = [c for c in c1 if avg_ovl(c, errs) <= ovl_th]
    print(f"   Ovl≤{ovl_th}: {len(c2)}")
    c3 = [c for c in c2 if combo_cost(c) <= cost_th]
    print(f"   Cost≤{cost_th}: {len(c3)}")
    scored = []
    for c in c3:
        s = prescore(c, perf_dict, errs, success_logs)
        scored.append({'combo': c, **s})
    scored.sort(key=lambda x: x['prescore'], reverse=True)
    return c1, c2, c3, scored, scored[:top_k]

def save_qualified_41(scored):
    rows = [{'Rank': i, 'Symbol': f'Q{i}',
             'Feature combination': ' + '.join(s['combo']),
             'Coverage': round(s['coverage'],4), 'BasePerf': round(s['baseperf'],4),
             'Com': round(s['com'],4), 'Ovl': round(s['ovl'],4),
             'Cost': round(s['cost'],4), 'PreScore': round(s['prescore'],4)}
            for i, s in enumerate(scored, 1)]
    df = pd.DataFrame(rows)
    df.to_csv(ensure_dir(SELECTION_DIR) / 'qualified_41.csv', index=False)
    return df

def save_top7(top):
    rows = [{'Rank': i, 'Symbol': f'S{i}',
             'Feature combination': ' + '.join(s['combo']),
             'Validation coverage': round(s['coverage'],4),
             'Com': round(s['com'],4), 'Ovl': round(s['ovl'],4),
             'Cost': round(s['cost'],4), 'PreScore': round(s['prescore'],4)}
            for i, s in enumerate(top, 1)]
    df = pd.DataFrame(rows)
    df.to_csv(ensure_dir(SELECTION_DIR) / 'top7.csv', index=False)
    return df