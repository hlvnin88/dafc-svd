#!/usr/bin/env python
"""Bước 16: Tái tạo bảng/hình từ results/*. Chỉ đọc, không sinh số."""
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import TABLES_DIR, SELECTION_DIR, print_config
from src.utils import print_section, print_table
from src.plotting import plot_pareto

def main():
    print_config()
    print_section("REPRODUCE — analysis only", '=')
    tables = sorted(TABLES_DIR.glob('*.csv'))
    if not tables:
        print("⚠️  Chưa có bảng nào. Chạy main.py trước."); return
    for p in tables:
        try:
            df = pd.read_csv(p)
            print_table(df, p.name, max_rows=20)
        except Exception as e:
            print(f"❌ {p.name}: {e}")

    top7 = SELECTION_DIR / 'top7.csv'
    if top7.exists():
        df = pd.read_csv(top7)
        info = [{'symbol': r['Symbol'], 'cost': float(r['Cost']),
                 'avg_f1': float(r['PreScore'])} for _, r in df.iterrows()]
        plot_pareto(info)
    print_section("✅ DONE", '=')

if __name__ == '__main__':
    main()