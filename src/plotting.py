"""Bước 16: Vẽ hình."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from config import FIGURES_DIR
from src.utils import ensure_dir

def plot_com_ovl_heatmaps(com_M, ovl_M, groups):
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    sns.heatmap(com_M, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=groups, yticklabels=groups,
                ax=ax[0], cbar_kws={'label':'Complementarity'})
    ax[0].set_title('(a) Complementarity Matrix (Com)')
    sns.heatmap(ovl_M, annot=True, fmt='.2f', cmap='Reds',
                xticklabels=groups, yticklabels=groups,
                ax=ax[1], cbar_kws={'label':'Overlap'})
    ax[1].set_title('(b) Error Overlap Matrix (Ovl)')
    plt.tight_layout()
    p = ensure_dir(FIGURES_DIR) / 'figure3_com_ovl.png'
    plt.savefig(p, dpi=200, bbox_inches='tight'); plt.close()
    print(f"💾 {p}")

def plot_funnel(stats):
    steps = list(stats.keys()); counts = list(stats.values())
    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.barh(steps, counts, color='skyblue', edgecolor='black')
    for b, v in zip(bars, counts):
        ax.text(v + 3, b.get_y()+b.get_height()/2, str(v), va='center')
    ax.set_xlabel('Số tổ hợp'); ax.set_title('Combination Filtering Funnel')
    plt.tight_layout()
    p = ensure_dir(FIGURES_DIR) / 'figure4_funnel.png'
    plt.savefig(p, dpi=200, bbox_inches='tight'); plt.close()
    print(f"💾 {p}")

def plot_pareto(info):
    fig, ax = plt.subplots(figsize=(8, 6))
    xs = [t['cost'] for t in info]; ys = [t['avg_f1'] for t in info]
    ax.scatter(xs, ys, s=120, c='green', edgecolors='black', zorder=3)
    for t in info:
        ax.annotate(t['symbol'], (t['cost'], t['avg_f1']),
                    textcoords='offset points', xytext=(0, 8),
                    ha='center', fontsize=9)
    ax.set_xlabel('Cost'); ax.set_ylabel('Average F1')
    ax.set_title('Pareto: Average F1 vs Cost')
    ax.grid(alpha=0.3); plt.tight_layout()
    p = ensure_dir(FIGURES_DIR) / 'figure4_pareto.png'
    plt.savefig(p, dpi=200, bbox_inches='tight'); plt.close()
    print(f"💾 {p}")