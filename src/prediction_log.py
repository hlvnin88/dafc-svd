"""Bước 5: Prediction / Error / Cost logs."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from config import (PRED_LOGS_DIR, ERR_VEC_DIR, COST_LOGS_DIR, TABLES_DIR,
                     COST_COMPONENT_WEIGHTS, RESOURCE_LOGS, FEATURE_GROUPS)
from src.utils import ensure_dir


def save_prediction_log(group, seed, y_true, probs, preds, sids):
    df = pd.DataFrame({'sample_id': sids, 'y_true': y_true.astype(int),
                        'probability': probs, 'y_pred': preds.astype(int),
                        'error': (preds.astype(int) != y_true.astype(int)).astype(int)})
    d = ensure_dir(PRED_LOGS_DIR / group)
    df.to_csv(d / f'seed{seed}.csv', index=False)


def save_error_vector(group, seed, errors):
    d = ensure_dir(ERR_VEC_DIR / group)
    np.save(d / f'seed{seed}.npy', errors.astype(np.int8))


def _fmt_time(sec):
    if sec < 3600: return f"{int(sec // 60)} min"
    return f"{sec / 3600:.2f} h"


def build_cost_components_table(dataset_name='devign'):
    """
    Đọc logs/resource/F*.csv → chuẩn hoá Min-Max → tính Cost (công thức 17) →
    ghi:
        results/tables/cost_components.csv    (Bảng #4)
        results/cost_logs/cost_log.csv        (cho metrics.py downstream)
    """
    rows = []
    for g in FEATURE_GROUPS:
        p = Path(RESOURCE_LOGS) / f'{g}.csv'
        if not p.exists():
            raise RuntimeError(f"Thiếu {p}. Chạy extract_features.py + individual_train trước.")
        df = pd.read_csv(p)
        df = df[df['dataset'] == dataset_name]
        if df.empty:
            raise RuntimeError(f"Không có dòng cho {dataset_name} trong {p}")
        # Lấy dòng cuối cùng (mới nhất)
        rows.append({
            'group': g,
            'extraction_time_sec': float(df['extraction_time'].iloc[-1]),
            'inference_time':      float(df['inference_time'].iloc[-1]),
            'dimension':           float(df['dimension'].iloc[-1]),
            'memory_mb':           float(df['memory_mb'].iloc[-1]),
            'failure_rate':        float(df['failure_rate'].iloc[-1]),
        })

    raw = pd.DataFrame(rows)
    for col in ['extraction_time_sec', 'inference_time',
                'dimension', 'memory_mb', 'failure_rate']:
        mn, mx = raw[col].min(), raw[col].max()
        rng = mx - mn if mx > mn else 1.0
        raw[col + '_n'] = (raw[col] - mn) / rng

    raw['cost'] = (COST_COMPONENT_WEIGHTS[0] * raw['extraction_time_sec_n'] +
                    COST_COMPONENT_WEIGHTS[1] * raw['inference_time_n'] +
                    COST_COMPONENT_WEIGHTS[2] * raw['dimension_n'] +
                    COST_COMPONENT_WEIGHTS[3] * raw['memory_mb_n'] +
                    COST_COMPONENT_WEIGHTS[4] * raw['failure_rate_n'])

    out_rows = [{
        'Group': r['group'],
        'Extraction time': _fmt_time(r['extraction_time_sec']),
        'Inference time (ms/sample)': round(r['inference_time'], 2),
        'Dimensionality': int(r['dimension']),
        'Peak memory (MB)': int(r['memory_mb']),
        'Failure rate (%)': round(r['failure_rate'], 2),
        'Cost': round(r['cost'], 3),
    } for _, r in raw.iterrows()]

    df_out = pd.DataFrame(out_rows)
    out_path = ensure_dir(TABLES_DIR) / 'cost_components.csv'
    df_out.to_csv(out_path, index=False)
    print(f"  💾 {out_path}")

    # cost_log.csv cho metrics.py
    cost_log = raw[['group', 'cost']].copy()
    cost_log.to_csv(ensure_dir(COST_LOGS_DIR) / 'cost_log.csv', index=False)
    return df_out


def save_valid_sample_ids(d):
    out = ensure_dir(COST_LOGS_DIR) / 'valid_sample_ids.json'
    with open(out, 'w') as f: json.dump(d, f, indent=2)
    return str(out)