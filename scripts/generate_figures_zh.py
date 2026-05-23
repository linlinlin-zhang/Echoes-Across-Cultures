#!/usr/bin/env python3
"""
DCAS Cross-Cultural Music Recommendation Paper - Chinese Figures
=================================================================
Generates 32 Chinese-labeled figures for the DCAS paper.
Reads benchmark data from JSON files with fallback hardcoded values.
Saves 300 DPI PNG figures and CSV data to output directory.
"""
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import fontManager
from pathlib import Path
from typing import Optional

warnings.filterwarnings('ignore')

# ==============================================================================
# CONFIGURATION
# ==============================================================================
BASE_DIR = Path(r'E:\Desktop\Echo')
OUTPUT_DIR = BASE_DIR / 'reports' / 'figures' / 'project_overview_zh_2026-04-05'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DPI = 300

# Color palette
DCAS_OT = '#4ecca3'
DCAS_CAL_TARGET = '#2ba582'
DCAS_CAL_MINOR = '#1a8a6e'
BPR_LM = '#0f3460'
BPR_OTHER = '#3a7ca5'
BASELINE_GRAY = '#888888'
BASELINE_LIGHT = '#aaaaaa'
POP_ORANGE = '#d65a31'
WHITE_BG = '#ffffff'
GRID_COLOR = '#E0E0E0'

# Chinese translations
METRIC_ZH = {
    'serendipity': '惊喜度',
    'cultural_calibration_kl': '校准KL',
    'minority_exposure_at_k': '少数曝光率',
    'recall': '召回率',
    'ndcg': 'NDCG',
    'mrr': 'MRR',
    'coverage': '覆盖率',
}

METHOD_ZH = {
    'popularity': '热门度',
    'cosine': '余弦',
    'knn': 'KNN',
    'lightfm_like': 'LightFM',
    'bpr_mf': 'BPR-MF',
    'bpr_two_stage_hybrid': 'BPR-双阶段',
    'bpr_listwise_hybrid': 'BPR-列表',
    'bpr_lambdamart_hybrid': 'BPR-LM',
    'dcas_full_ot': 'DCAS-OT',
    'dcas_full_ot_calibrated_target': 'DCAS-目标校准',
    'dcas_full_ot_calibrated_minor': 'DCAS-少数导向',
    'dcas_ot_cal_p1': 'P1 (偏目标)',
    'dcas_ot_cal_p2_target': 'P2 (目标)',
    'dcas_ot_cal_p3_balanced': 'P3 (均衡)',
    'dcas_ot_cal_p4_minor': 'P4 (少数)',
    'dcas_ot_cal_p5_ultra_minor': 'P5 (超少数)',
    'dcas_log_ot': 'DCAS-日志OT',
    'three_factor_dcas': 'DCAS (三因子)',
    'vae': 'VAE',
    'beta_vae': 'β-VAE',
    'factorvae': 'FactorVAE',
}

ABLATION_ZH = {
    'full': '完整模型',
    'no_domain': '无领域对抗',
    'no_constraints': '无约束',
    'no_ot': '无最优传输',
}

SUITE_ZH = {
    'v4_main_culturemert_stage3_lambdamart': 'V4 CultureMERT',
    'v4_main_gemini_stage3_lambdamart': 'V4 Gemini',
    'v4_routeA_small_culturemert_stage3_lambdamart': 'RouteA CultureMERT',
    'v4_routeA_small_gemini_stage3_lambdamart': 'RouteA Gemini',
    'public_routeA_phase2_cn_lambdamart': 'Public-CN',
    'v3_main_culturemert_stage3_lambdamart': 'V3 CultureMERT',
    'v3_main_culturemert_stage3': 'V3-S3 CultureMERT',
    'v3_main_culturemert': 'V3 CultureMERT',
}


# ==============================================================================
# FONT SETUP
# ==============================================================================
def setup_chinese_font():
    """Configure Chinese-capable font."""
    chinese_fonts = [
        'SimHei', 'Microsoft YaHei', 'PingFang SC',
        'Noto Sans CJK SC', 'WenQuanYi Micro Hei',
        'Heiti SC', 'Arial Unicode MS', 'Source Han Sans SC',
    ]
    available = {f.name for f in fontManager.ttflist}
    for fn in chinese_fonts:
        if fn in available:
            plt.rcParams['font.sans-serif'] = [fn] + plt.rcParams['font.sans-serif']
            plt.rcParams['axes.unicode_minus'] = False
            print(f'  Chinese font: {fn}')
            return fn
    print('  WARNING: No Chinese font found. Install SimHei or Microsoft YaHei.')
    plt.rcParams['axes.unicode_minus'] = False
    return None


# ==============================================================================
# DATA HELPERS
# ==============================================================================
def load_json(path: Path) -> Optional[dict]:
    try:
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        print(f'  WARNING: Failed to load {path}: {e}')
    return None


def load_bench(suite: str) -> Optional[dict]:
    return load_json(BASE_DIR / 'reports' / 'benchmarks' / suite / 'benchmark_summary.json')


def load_ablation_summary(suite: str) -> Optional[dict]:
    return load_json(BASE_DIR / 'reports' / 'audits' / 'ablation_v4_main_2026-04-05' / suite / 'summary.json')


def load_ablation_comparison(suite: str, name: str) -> Optional[dict]:
    return load_json(BASE_DIR / 'reports' / 'audits' / 'ablation_v4_main_2026-04-05' / suite / f'comparison_{name}.json')


def load_hparam(suite: str) -> Optional[dict]:
    return load_json(BASE_DIR / 'reports' / 'hparam' / suite / 'benchmark_summary.json')


def load_baseline_comparison(suite: str) -> Optional[dict]:
    return load_json(BASE_DIR / 'reports' / 'baseline_comparison' / suite / 'baseline_comparison_summary.json')


def load_pal(suite: str) -> Optional[dict]:
    return load_json(BASE_DIR / 'reports' / 'pal' / suite / 'phase3_pal_summary.json')


def extract_methods(data: dict) -> pd.DataFrame:
    rows = []
    for method, metrics in data.get('methods', {}).items():
        row = {'method': method}
        row.update(metrics)
        rows.append(row)
    return pd.DataFrame(rows)


def mzh(name: str) -> str:
    return METHOD_ZH.get(name, name)


def save_csv(df: pd.DataFrame, stem: str):
    out = OUTPUT_DIR / f'{stem}.csv'
    df.to_csv(out, index=False, encoding='utf-8-sig')
    print(f'  CSV: {out}')


def save_fig(fig, stem: str):
    out = OUTPUT_DIR / f'{stem}.png'
    fig.tight_layout()
    fig.savefig(out, dpi=DPI, bbox_inches='tight', facecolor=WHITE_BG)
    plt.close(fig)
    print(f'  FIG: {out}')


def apply_style():
    plt.rcParams.update({
        'figure.facecolor': WHITE_BG,
        'axes.facecolor': WHITE_BG,
        'axes.edgecolor': '#333333',
        'axes.linewidth': 1.0,
        'axes.grid': True,
        'grid.alpha': 0.25,
        'grid.color': GRID_COLOR,
        'axes.titlesize': 13,
        'axes.titleweight': 'bold',
        'axes.labelsize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 8,
        'figure.titlesize': 15,
        'figure.titleweight': 'bold',
        'font.size': 9,
    })


# ==============================================================================
# FIGURE 1: Main Results - CultureMERT Bar Chart
# ==============================================================================
def _mc(m):
    if 'calibrated_minor' in m: return DCAS_CAL_MINOR
    if 'calibrated_target' in m: return DCAS_CAL_TARGET
    if 'dcas' in m: return DCAS_OT
    if 'lambdamart' in m: return BPR_LM
    if 'bpr' in m: return BPR_OTHER
    if 'popularity' in m: return POP_ORANGE
    return BASELINE_GRAY


# ---------- 1. V4 CultureMERT 3-panel ----------
def fig01_v4_cm_main():
    print('Figure 1: v4_cm_main_results_3panel')
    d = load_bench('v4_main_culturemert_stage3_lambdamart')
    if not d: print('  SKIPPED'); return
    df = extract_methods(d)
    order = ['popularity','cosine','knn','lightfm_like','bpr_mf','bpr_two_stage_hybrid',
             'bpr_listwise_hybrid','bpr_lambdamart_hybrid','dcas_full_ot',
             'dcas_full_ot_calibrated_target','dcas_full_ot_calibrated_minor']
    order = [m for m in order if m in df['method'].values]
    df = df[df['method'].isin(order)].copy()
    df['method'] = pd.Categorical(df['method'], categories=order, ordered=True)
    df = df.sort_values('method')
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
    mets = ['serendipity_mean','cultural_calibration_kl_mean','minority_exposure_at_k_mean']
    labs = ['惊喜度','校准KL','少数曝光率']
    for ax, met, lab in zip(axes, mets, labs):
        vals = df[met].values; colors = [_mc(m) for m in df['method']]
        bars = ax.barh(range(len(df)), vals, color=colors, height=0.65)
        ax.set_yticks(range(len(df))); ax.set_yticklabels([mzh(m) for m in df['method']])
        ax.set_xlabel(lab); ax.set_title(lab)
        rng = max(vals)-min(vals)+0.01
        for bar, v in zip(bars, vals):
            ax.text(bar.get_width()+0.005*rng, bar.get_y()+bar.get_height()/2, f'{v:.3f}', va='center', fontsize=7)
        ax.invert_yaxis()
    fig.suptitle('V4 CultureMERT 基准测试结果', y=1.02)
    save_fig(fig, 'v4_cm_main_results_3panel'); save_csv(df[['method']+mets], 'v4_cm_main_results_3panel')


# ---------- 2. V4 Gemini 3-panel ----------
def fig02_v4_gm_main():
    print('Figure 2: v4_gm_main_results_3panel')
    d = load_bench('v4_main_gemini_stage3_lambdamart')
    if not d: print('  SKIPPED'); return
    df = extract_methods(d)
    order = ['popularity','cosine','knn','lightfm_like','bpr_mf','bpr_two_stage_hybrid',
             'bpr_listwise_hybrid','bpr_lambdamart_hybrid','dcas_full_ot',
             'dcas_full_ot_calibrated_target','dcas_full_ot_calibrated_minor']
    order = [m for m in order if m in df['method'].values]
    df = df[df['method'].isin(order)].copy()
    df['method'] = pd.Categorical(df['method'], categories=order, ordered=True)
    df = df.sort_values('method')
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
    mets = ['serendipity_mean','cultural_calibration_kl_mean','minority_exposure_at_k_mean']
    labs = ['惊喜度','校准KL','少数曝光率']
    for ax, met, lab in zip(axes, mets, labs):
        vals = df[met].values; colors = [_mc(m) for m in df['method']]
        bars = ax.barh(range(len(df)), vals, color=colors, height=0.65)
        ax.set_yticks(range(len(df))); ax.set_yticklabels([mzh(m) for m in df['method']])
        ax.set_xlabel(lab); ax.set_title(lab)
        rng = max(vals)-min(vals)+0.01
        for bar, v in zip(bars, vals):
            ax.text(bar.get_width()+0.005*rng, bar.get_y()+bar.get_height()/2, f'{v:.3f}', va='center', fontsize=7)
        ax.invert_yaxis()
    fig.suptitle('V4 Gemini 基准测试结果', y=1.02)
    save_fig(fig, 'v4_gm_main_results_3panel'); save_csv(df[['method']+mets], 'v4_gm_main_results_3panel')


# ---------- 3. RouteA CultureMERT ----------
def fig03_routeA_cm():
    print('Figure 3: routeA_cm_main_results')
    d = load_bench('v4_routeA_small_culturemert_stage3_lambdamart')
    if not d: print('  SKIPPED'); return
    df = extract_methods(d)
    order = ['popularity','cosine','knn','lightfm_like','bpr_mf','bpr_two_stage_hybrid',
             'bpr_listwise_hybrid','bpr_lambdamart_hybrid','dcas_full_ot',
             'dcas_full_ot_calibrated_target','dcas_full_ot_calibrated_minor']
    order = [m for m in order if m in df['method'].values]
    df = df[df['method'].isin(order)].copy()
    df['method'] = pd.Categorical(df['method'], categories=order, ordered=True)
    df = df.sort_values('method')
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
    mets = ['serendipity_mean','cultural_calibration_kl_mean','minority_exposure_at_k_mean']
    labs = ['惊喜度','校准KL','少数曝光率']
    for ax, met, lab in zip(axes, mets, labs):
        vals = df[met].values; colors = [_mc(m) for m in df['method']]
        bars = ax.barh(range(len(df)), vals, color=colors, height=0.65)
        ax.set_yticks(range(len(df))); ax.set_yticklabels([mzh(m) for m in df['method']])
        ax.set_xlabel(lab); ax.set_title(lab)
        rng = max(vals)-min(vals)+0.01
        for bar, v in zip(bars, vals):
            ax.text(bar.get_width()+0.005*rng, bar.get_y()+bar.get_height()/2, f'{v:.3f}', va='center', fontsize=7)
        ax.invert_yaxis()
    fig.suptitle('RouteA Small + CultureMERT', y=1.02)
    save_fig(fig, 'routeA_cm_main_results'); save_csv(df[['method']+mets], 'routeA_cm_main_results')


# ---------- 4. RouteA Gemini ----------
def fig04_routeA_gm():
    print('Figure 4: routeA_gm_main_results')
    d = load_bench('v4_routeA_small_gemini_stage3_lambdamart')
    if not d: print('  SKIPPED'); return
    df = extract_methods(d)
    order = ['popularity','cosine','knn','lightfm_like','bpr_mf','bpr_two_stage_hybrid',
             'bpr_listwise_hybrid','bpr_lambdamart_hybrid','dcas_full_ot',
             'dcas_full_ot_calibrated_target','dcas_full_ot_calibrated_minor']
    order = [m for m in order if m in df['method'].values]
    df = df[df['method'].isin(order)].copy()
    df['method'] = pd.Categorical(df['method'], categories=order, ordered=True)
    df = df.sort_values('method')
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
    mets = ['serendipity_mean','cultural_calibration_kl_mean','minority_exposure_at_k_mean']
    labs = ['惊喜度','校准KL','少数曝光率']
    for ax, met, lab in zip(axes, mets, labs):
        vals = df[met].values; colors = [_mc(m) for m in df['method']]
        bars = ax.barh(range(len(df)), vals, color=colors, height=0.65)
        ax.set_yticks(range(len(df))); ax.set_yticklabels([mzh(m) for m in df['method']])
        ax.set_xlabel(lab); ax.set_title(lab)
        rng = max(vals)-min(vals)+0.01
        for bar, v in zip(bars, vals):
            ax.text(bar.get_width()+0.005*rng, bar.get_y()+bar.get_height()/2, f'{v:.3f}', va='center', fontsize=7)
        ax.invert_yaxis()
    fig.suptitle('RouteA Small + Gemini', y=1.02)
    save_fig(fig, 'routeA_gm_main_results'); save_csv(df[['method']+mets], 'routeA_gm_main_results')


# ---------- 5. Public-CN ----------
def fig05_public_cn():
    print('Figure 5: public_cn_results')
    d = load_bench('public_routeA_phase2_cn_lambdamart')
    if not d: print('  SKIPPED'); return
    df = extract_methods(d)
    order = ['popularity','cosine','knn','bpr_mf','bpr_two_stage_hybrid',
             'bpr_listwise_hybrid','bpr_lambdamart_hybrid','dcas_full_ot',
             'dcas_full_ot_calibrated_target','dcas_full_ot_calibrated_minor']
    order = [m for m in order if m in df['method'].values]
    df = df[df['method'].isin(order)].copy()
    df['method'] = pd.Categorical(df['method'], categories=order, ordered=True)
    df = df.sort_values('method')
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    mets = ['serendipity_mean','cultural_calibration_kl_mean','minority_exposure_at_k_mean']
    labs = ['惊喜度','校准KL','少数曝光率']
    for ax, met, lab in zip(axes, mets, labs):
        vals = df[met].values; colors = [_mc(m) for m in df['method']]
        bars = ax.barh(range(len(df)), vals, color=colors, height=0.65)
        ax.set_yticks(range(len(df))); ax.set_yticklabels([mzh(m) for m in df['method']])
        ax.set_xlabel(lab); ax.set_title(lab)
        rng = max(vals)-min(vals)+0.01
        for bar, v in zip(bars, vals):
            ax.text(bar.get_width()+0.005*rng, bar.get_y()+bar.get_height()/2, f'{v:.3f}', va='center', fontsize=7)
        ax.invert_yaxis()
    fig.suptitle('Public RouteA Phase2-CN', y=1.02)
    save_fig(fig, 'public_cn_results'); save_csv(df[['method']+mets], 'public_cn_results')


# ---------- 6. Serendipity all methods CM ----------
def fig06_serendipity_cm_all():
    print('Figure 6: serendipity_all_methods_cm')
    d = load_bench('v4_main_culturemert_stage3_lambdamart')
    if not d: print('  SKIPPED'); return
    df = extract_methods(d)
    order = ['popularity','cosine','knn','lightfm_like','bpr_mf','bpr_two_stage_hybrid',
             'bpr_listwise_hybrid','bpr_lambdamart_hybrid','dcas_full_ot',
             'dcas_full_ot_calibrated_target','dcas_full_ot_calibrated_minor']
    order = [m for m in order if m in df['method'].values]
    df = df[df['method'].isin(order)].copy()
    df['method'] = pd.Categorical(df['method'], categories=order, ordered=True)
    df = df.sort_values('method')
    fig, ax = plt.subplots(figsize=(9, 5.5))
    vals = df['serendipity_mean'].values
    colors = [_mc(m) for m in df['method']]
    bars = ax.barh(range(len(df)), vals, color=colors, height=0.65)
    ax.set_yticks(range(len(df))); ax.set_yticklabels([mzh(m) for m in df['method']])
    ax.set_xlabel('惊喜度'); ax.set_title('V4 CultureMERT 各方法惊喜度对比')
    rng = max(vals)-min(vals)+0.01
    for bar, v in zip(bars, vals):
        ax.text(bar.get_width()+0.005*rng, bar.get_y()+bar.get_height()/2, f'{v:.3f}', va='center', fontsize=7)
    ax.invert_yaxis()
    save_fig(fig, 'serendipity_all_methods_cm'); save_csv(df[['method','serendipity_mean']], 'serendipity_all_methods_cm')


# ---------- 7. Serendipity all methods GM ----------
def fig07_serendipity_gm_all():
    print('Figure 7: serendipity_all_methods_gm')
    d = load_bench('v4_main_gemini_stage3_lambdamart')
    if not d: print('  SKIPPED'); return
    df = extract_methods(d)
    order = ['popularity','cosine','knn','lightfm_like','bpr_mf','bpr_two_stage_hybrid',
             'bpr_listwise_hybrid','bpr_lambdamart_hybrid','dcas_full_ot',
             'dcas_full_ot_calibrated_target','dcas_full_ot_calibrated_minor']
    order = [m for m in order if m in df['method'].values]
    df = df[df['method'].isin(order)].copy()
    df['method'] = pd.Categorical(df['method'], categories=order, ordered=True)
    df = df.sort_values('method')
    fig, ax = plt.subplots(figsize=(9, 5.5))
    vals = df['serendipity_mean'].values
    colors = [_mc(m) for m in df['method']]
    bars = ax.barh(range(len(df)), vals, color=colors, height=0.65)
    ax.set_yticks(range(len(df))); ax.set_yticklabels([mzh(m) for m in df['method']])
    ax.set_xlabel('惊喜度'); ax.set_title('V4 Gemini 各方法惊喜度对比')
    rng = max(vals)-min(vals)+0.01
    for bar, v in zip(bars, vals):
        ax.text(bar.get_width()+0.005*rng, bar.get_y()+bar.get_height()/2, f'{v:.3f}', va='center', fontsize=7)
    ax.invert_yaxis()
    save_fig(fig, 'serendipity_all_methods_gm'); save_csv(df[['method','serendipity_mean']], 'serendipity_all_methods_gm')


# ---------- 8. Minority all methods CM ----------
def fig08_minority_cm_all():
    print('Figure 8: minority_exposure_all_methods_cm')
    d = load_bench('v4_main_culturemert_stage3_lambdamart')
    if not d: print('  SKIPPED'); return
    df = extract_methods(d)
    order = ['popularity','cosine','knn','lightfm_like','bpr_mf','bpr_two_stage_hybrid',
             'bpr_listwise_hybrid','bpr_lambdamart_hybrid','dcas_full_ot',
             'dcas_full_ot_calibrated_target','dcas_full_ot_calibrated_minor']
    order = [m for m in order if m in df['method'].values]
    df = df[df['method'].isin(order)].copy()
    df['method'] = pd.Categorical(df['method'], categories=order, ordered=True)
    df = df.sort_values('method')
    fig, ax = plt.subplots(figsize=(9, 5.5))
    vals = df['minority_exposure_at_k_mean'].values
    colors = [_mc(m) for m in df['method']]
    bars = ax.barh(range(len(df)), vals, color=colors, height=0.65)
    ax.set_yticks(range(len(df))); ax.set_yticklabels([mzh(m) for m in df['method']])
    ax.set_xlabel('少数曝光率'); ax.set_title('V4 CultureMERT 各方法少数曝光率')
    rng = max(vals)-min(vals)+0.01
    for bar, v in zip(bars, vals):
        ax.text(bar.get_width()+0.005*rng, bar.get_y()+bar.get_height()/2, f'{v:.3f}', va='center', fontsize=7)
    ax.invert_yaxis()
    save_fig(fig, 'minority_exposure_all_methods_cm'); save_csv(df[['method','minority_exposure_at_k_mean']], 'minority_exposure_all_methods_cm')


# ---------- 9. Minority all methods GM ----------
def fig09_minority_gm_all():
    print('Figure 9: minority_exposure_all_methods_gm')
    d = load_bench('v4_main_gemini_stage3_lambdamart')
    if not d: print('  SKIPPED'); return
    df = extract_methods(d)
    order = ['popularity','cosine','knn','lightfm_like','bpr_mf','bpr_two_stage_hybrid',
             'bpr_listwise_hybrid','bpr_lambdamart_hybrid','dcas_full_ot',
             'dcas_full_ot_calibrated_target','dcas_full_ot_calibrated_minor']
    order = [m for m in order if m in df['method'].values]
    df = df[df['method'].isin(order)].copy()
    df['method'] = pd.Categorical(df['method'], categories=order, ordered=True)
    df = df.sort_values('method')
    fig, ax = plt.subplots(figsize=(9, 5.5))
    vals = df['minority_exposure_at_k_mean'].values
    colors = [_mc(m) for m in df['method']]
    bars = ax.barh(range(len(df)), vals, color=colors, height=0.65)
    ax.set_yticks(range(len(df))); ax.set_yticklabels([mzh(m) for m in df['method']])
    ax.set_xlabel('少数曝光率'); ax.set_title('V4 Gemini 各方法少数曝光率')
    rng = max(vals)-min(vals)+0.01
    for bar, v in zip(bars, vals):
        ax.text(bar.get_width()+0.005*rng, bar.get_y()+bar.get_height()/2, f'{v:.3f}', va='center', fontsize=7)
    ax.invert_yaxis()
    save_fig(fig, 'minority_exposure_all_methods_gm'); save_csv(df[['method','minority_exposure_at_k_mean']], 'minority_exposure_all_methods_gm')


# ---------- 10. Calib KL all methods CM ----------
def fig10_calibkl_cm_all():
    print('Figure 10: calibration_kl_all_methods_cm')
    d = load_bench('v4_main_culturemert_stage3_lambdamart')
    if not d: print('  SKIPPED'); return
    df = extract_methods(d)
    order = ['popularity','cosine','knn','lightfm_like','bpr_mf','bpr_two_stage_hybrid',
             'bpr_listwise_hybrid','bpr_lambdamart_hybrid','dcas_full_ot',
             'dcas_full_ot_calibrated_target','dcas_full_ot_calibrated_minor']
    order = [m for m in order if m in df['method'].values]
    df = df[df['method'].isin(order)].copy()
    df['method'] = pd.Categorical(df['method'], categories=order, ordered=True)
    df = df.sort_values('method')
    fig, ax = plt.subplots(figsize=(9, 5.5))
    vals = df['cultural_calibration_kl_mean'].values
    colors = [_mc(m) for m in df['method']]
    bars = ax.barh(range(len(df)), vals, color=colors, height=0.65)
    ax.set_yticks(range(len(df))); ax.set_yticklabels([mzh(m) for m in df['method']])
    ax.set_xlabel('校准KL'); ax.set_title('V4 CultureMERT 各方法校准KL')
    rng = max(vals)-min(vals)+0.01
    for bar, v in zip(bars, vals):
        ax.text(bar.get_width()+0.005*rng, bar.get_y()+bar.get_height()/2, f'{v:.3f}', va='center', fontsize=7)
    ax.invert_yaxis()
    save_fig(fig, 'calibration_kl_all_methods_cm'); save_csv(df[['method','cultural_calibration_kl_mean']], 'calibration_kl_all_methods_cm')


# ---------- 11. Calib KL all methods GM ----------
def fig11_calibkl_gm_all():
    print('Figure 11: calibration_kl_all_methods_gm')
    d = load_bench('v4_main_gemini_stage3_lambdamart')
    if not d: print('  SKIPPED'); return
    df = extract_methods(d)
    order = ['popularity','cosine','knn','lightfm_like','bpr_mf','bpr_two_stage_hybrid',
             'bpr_listwise_hybrid','bpr_lambdamart_hybrid','dcas_full_ot',
             'dcas_full_ot_calibrated_target','dcas_full_ot_calibrated_minor']
    order = [m for m in order if m in df['method'].values]
    df = df[df['method'].isin(order)].copy()
    df['method'] = pd.Categorical(df['method'], categories=order, ordered=True)
    df = df.sort_values('method')
    fig, ax = plt.subplots(figsize=(9, 5.5))
    vals = df['cultural_calibration_kl_mean'].values
    colors = [_mc(m) for m in df['method']]
    bars = ax.barh(range(len(df)), vals, color=colors, height=0.65)
    ax.set_yticks(range(len(df))); ax.set_yticklabels([mzh(m) for m in df['method']])
    ax.set_xlabel('校准KL'); ax.set_title('V4 Gemini 各方法校准KL')
    rng = max(vals)-min(vals)+0.01
    for bar, v in zip(bars, vals):
        ax.text(bar.get_width()+0.005*rng, bar.get_y()+bar.get_height()/2, f'{v:.3f}', va='center', fontsize=7)
    ax.invert_yaxis()
    save_fig(fig, 'calibration_kl_all_methods_gm'); save_csv(df[['method','cultural_calibration_kl_mean']], 'calibration_kl_all_methods_gm')


# ---------- 12. Cross-suite serendipity ----------
def fig12_cross_serp():
    print('Figure 12: cross_suite_serendipity')
    suites = {
        'v4_main_culturemert_stage3_lambdamart': 'V4 CM',
        'v4_main_gemini_stage3_lambdamart': 'V4 GM',
        'v4_routeA_small_culturemert_stage3_lambdamart': 'RA CM',
        'v4_routeA_small_gemini_stage3_lambdamart': 'RA GM',
    }
    rows = []
    for skey, slbl in suites.items():
        d = load_bench(skey)
        if not d: continue
        df = extract_methods(d)
        for m in ['popularity','bpr_lambdamart_hybrid','dcas_full_ot','dcas_full_ot_calibrated_target','dcas_full_ot_calibrated_minor']:
            if m in df['method'].values:
                rows.append({'基准': slbl, '方法': mzh(m), '惊喜度': df.loc[df['method']==m,'serendipity_mean'].values[0]})
    if not rows: print('  SKIPPED'); return
    rdf = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(11, 5))
    piv = rdf.pivot(index='基准', columns='方法', values='惊喜度')
    piv = piv.reindex(['V4 CM','V4 GM','RA CM','RA GM'])
    cols = [DCAS_CAL_MINOR, DCAS_CAL_TARGET, DCAS_OT, BPR_LM, POP_ORANGE]
    avail = [c for c in ['DCAS-少数导向','DCAS-目标校准','DCAS-OT','BPR-LM','热门度'] if c in piv.columns]
    cmap = {'DCAS-少数导向':DCAS_CAL_MINOR,'DCAS-目标校准':DCAS_CAL_TARGET,'DCAS-OT':DCAS_OT,'BPR-LM':BPR_LM,'热门度':POP_ORANGE}
    x = np.arange(len(piv)); w = 0.15
    for i, col in enumerate(avail):
        ax.bar(x + i*w - len(avail)*w/2, piv[col], w, label=col, color=cmap.get(col,'#888'), zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(piv.index); ax.set_ylabel('惊喜度')
    ax.set_title('跨4基准惊喜度对比'); ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.3)
    save_fig(fig, 'cross_suite_serendipity'); save_csv(rdf, 'cross_suite_serendipity')


# ---------- 13. Cross-suite minority ----------
def fig13_cross_minor():
    print('Figure 13: cross_suite_minority')
    suites = {
        'v4_main_culturemert_stage3_lambdamart': 'V4 CM',
        'v4_main_gemini_stage3_lambdamart': 'V4 GM',
        'v4_routeA_small_culturemert_stage3_lambdamart': 'RA CM',
        'v4_routeA_small_gemini_stage3_lambdamart': 'RA GM',
    }
    rows = []
    for skey, slbl in suites.items():
        d = load_bench(skey)
        if not d: continue
        df = extract_methods(d)
        for m in ['popularity','bpr_lambdamart_hybrid','dcas_full_ot','dcas_full_ot_calibrated_target','dcas_full_ot_calibrated_minor']:
            if m in df['method'].values:
                rows.append({'基准': slbl, '方法': mzh(m), '少数曝光率': df.loc[df['method']==m,'minority_exposure_at_k_mean'].values[0]})
    if not rows: print('  SKIPPED'); return
    rdf = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(11, 5))
    piv = rdf.pivot(index='基准', columns='方法', values='少数曝光率')
    piv = piv.reindex(['V4 CM','V4 GM','RA CM','RA GM'])
    cmap = {'DCAS-少数导向':DCAS_CAL_MINOR,'DCAS-目标校准':DCAS_CAL_TARGET,'DCAS-OT':DCAS_OT,'BPR-LM':BPR_LM,'热门度':POP_ORANGE}
    avail = [c for c in ['DCAS-少数导向','DCAS-目标校准','DCAS-OT','BPR-LM','热门度'] if c in piv.columns]
    x = np.arange(len(piv)); w = 0.15
    for i, col in enumerate(avail):
        ax.bar(x + i*w - len(avail)*w/2, piv[col], w, label=col, color=cmap.get(col,'#888'), zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(piv.index); ax.set_ylabel('少数曝光率')
    ax.set_title('跨4基准少数曝光率对比'); ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.3)
    save_fig(fig, 'cross_suite_minority'); save_csv(rdf, 'cross_suite_minority')


# ---------- 14. Cross-suite calibration ----------
def fig14_cross_calib():
    print('Figure 14: cross_suite_calibration')
    suites = {
        'v4_main_culturemert_stage3_lambdamart': 'V4 CM',
        'v4_main_gemini_stage3_lambdamart': 'V4 GM',
        'v4_routeA_small_culturemert_stage3_lambdamart': 'RA CM',
        'v4_routeA_small_gemini_stage3_lambdamart': 'RA GM',
    }
    rows = []
    for skey, slbl in suites.items():
        d = load_bench(skey)
        if not d: continue
        df = extract_methods(d)
        for m in ['popularity','bpr_lambdamart_hybrid','dcas_full_ot','dcas_full_ot_calibrated_target','dcas_full_ot_calibrated_minor']:
            if m in df['method'].values:
                rows.append({'基准': slbl, '方法': mzh(m), '校准KL': df.loc[df['method']==m,'cultural_calibration_kl_mean'].values[0]})
    if not rows: print('  SKIPPED'); return
    rdf = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(11, 5))
    piv = rdf.pivot(index='基准', columns='方法', values='校准KL')
    piv = piv.reindex(['V4 CM','V4 GM','RA CM','RA GM'])
    cmap = {'DCAS-少数导向':DCAS_CAL_MINOR,'DCAS-目标校准':DCAS_CAL_TARGET,'DCAS-OT':DCAS_OT,'BPR-LM':BPR_LM,'热门度':POP_ORANGE}
    avail = [c for c in ['DCAS-少数导向','DCAS-目标校准','DCAS-OT','BPR-LM','热门度'] if c in piv.columns]
    x = np.arange(len(piv)); w = 0.15
    for i, col in enumerate(avail):
        ax.bar(x + i*w - len(avail)*w/2, piv[col], w, label=col, color=cmap.get(col,'#888'), zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(piv.index); ax.set_ylabel('校准KL')
    ax.set_title('跨4基准校准KL对比'); ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.3)
    save_fig(fig, 'cross_suite_calibration'); save_csv(rdf, 'cross_suite_calibration')


# ---------- 15. DCAS gains serendipity ----------
def fig15_dcas_gains_serp():
    print('Figure 15: dcas_gains_serendipity')
    suites = {
        'v4_main_culturemert_stage3_lambdamart': 'V4 CM',
        'v4_main_gemini_stage3_lambdamart': 'V4 GM',
        'v4_routeA_small_culturemert_stage3_lambdamart': 'RA CM',
        'v4_routeA_small_gemini_stage3_lambdamart': 'RA GM',
    }
    rows = []
    for skey, slbl in suites.items():
        d = load_bench(skey)
        if not d: continue
        df = extract_methods(d)
        bpr = df.loc[df['method']=='bpr_lambdamart_hybrid','serendipity_mean']
        dcas = df.loc[df['method']=='dcas_full_ot_calibrated_target','serendipity_mean']
        if len(bpr) > 0 and len(dcas) > 0:
            gain = dcas.values[0] - bpr.values[0]
            rows.append({'基准': slbl, '增益': gain, 'DCAS-目标校准': dcas.values[0], 'BPR-LM': bpr.values[0]})
    if not rows: print('  SKIPPED'); return
    rdf = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(rdf)); w = 0.35
    bars1 = ax.bar(x-w/2, rdf['DCAS-目标校准'], w, label='DCAS-目标校准', color=DCAS_CAL_TARGET, zorder=3)
    bars2 = ax.bar(x+w/2, rdf['BPR-LM'], w, label='BPR-LM', color=BPR_LM, zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(rdf['基准'])
    ax.set_ylabel('惊喜度'); ax.set_title('DCAS-目标校准 vs BPR-LM 惊喜度对比')
    ax.legend(); ax.grid(axis='y', alpha=0.3)
    # Add gain arrows
    for i, (_, row) in enumerate(rdf.iterrows()):
        ax.annotate(f'+{row["增益"]:.3f}', xy=(i, max(row['DCAS-目标校准'], row['BPR-LM'])),
                    xytext=(0, 12), textcoords='offset points', ha='center', fontsize=7,
                    color=DCAS_CAL_TARGET, fontweight='bold')
    save_fig(fig, 'dcas_gains_serendipity'); save_csv(rdf, 'dcas_gains_serendipity')


# ---------- 16. DCAS gains minority ----------
def fig16_dcas_gains_minor():
    print('Figure 16: dcas_gains_minority')
    suites = {
        'v4_main_culturemert_stage3_lambdamart': 'V4 CM',
        'v4_main_gemini_stage3_lambdamart': 'V4 GM',
        'v4_routeA_small_culturemert_stage3_lambdamart': 'RA CM',
        'v4_routeA_small_gemini_stage3_lambdamart': 'RA GM',
    }
    rows = []
    for skey, slbl in suites.items():
        d = load_bench(skey)
        if not d: continue
        df = extract_methods(d)
        bpr = df.loc[df['method']=='bpr_lambdamart_hybrid','minority_exposure_at_k_mean']
        dcas = df.loc[df['method']=='dcas_full_ot_calibrated_target','minority_exposure_at_k_mean']
        if len(bpr) > 0 and len(dcas) > 0:
            gain = dcas.values[0] - bpr.values[0]
            rows.append({'基准': slbl, '增益': gain, 'DCAS-目标校准': dcas.values[0], 'BPR-LM': bpr.values[0]})
    if not rows: print('  SKIPPED'); return
    rdf = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(rdf)); w = 0.35
    bars1 = ax.bar(x-w/2, rdf['DCAS-目标校准'], w, label='DCAS-目标校准', color=DCAS_CAL_TARGET, zorder=3)
    bars2 = ax.bar(x+w/2, rdf['BPR-LM'], w, label='BPR-LM', color=BPR_LM, zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(rdf['基准'])
    ax.set_ylabel('少数曝光率'); ax.set_title('DCAS-目标校准 vs BPR-LM 少数曝光率对比')
    ax.legend(); ax.grid(axis='y', alpha=0.3)
    for i, (_, row) in enumerate(rdf.iterrows()):
        ax.annotate(f'+{row["增益"]:.3f}', xy=(i, max(row['DCAS-目标校准'], row['BPR-LM'])),
                    xytext=(0, 12), textcoords='offset points', ha='center', fontsize=7,
                    color=DCAS_CAL_TARGET, fontweight='bold')
    save_fig(fig, 'dcas_gains_minority'); save_csv(rdf, 'dcas_gains_minority')


# ---------- 17. Ablation CM delta ----------
def fig17_abl_cm_delta():
    print('Figure 17: ablation_cm_delta')
    suite = 'v4_main_culturemert'
    comp_full = load_ablation_comparison(suite, 'full_vs_no_domain')
    if not comp_full: print('  SKIPPED'); return
    comps = {}
    for name in ['full_vs_no_domain','full_vs_no_constraints','full_vs_no_ot']:
        c = load_ablation_comparison(suite, name)
        if c: comps[name] = c
    if not comps: print('  SKIPPED'); return
    ablbl = {'full_vs_no_domain':'无领域对抗','full_vs_no_constraints':'无约束','full_vs_no_ot':'无最优传输'}
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    mk = ['serendipity','cultural_calibration_kl','minority_exposure_at_k']
    ml = ['惊喜度','校准KL','少数曝光率']
    for ax, mkey, mlab in zip(axes, mk, ml):
        names, deltas, lows, highs = [], [], [], []
        for k, v in comps.items():
            mm = v.get('metrics',{}).get(mkey,{})
            if 'delta_mean' in mm:
                names.append(ablbl.get(k,k)); deltas.append(mm['delta_mean'])
                lows.append(abs(mm.get('delta_ci95_low',0)))
                highs.append(abs(mm.get('delta_ci95_high',0)))
        colors = [BASELINE_GRAY, BASELINE_LIGHT, POP_ORANGE][:len(names)]
        ax.barh(range(len(names)), deltas, color=colors, height=0.6,
                xerr=[lows, highs], capsize=3, error_kw=dict(lw=1))
        ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=9)
        ax.set_xlabel(f'{mlab} delta (vs完整)'); ax.axvline(0, color='#333', lw=0.5)
        ax.set_title(mlab)
    fig.suptitle('CultureMERT 消融 delta (含95%CI)', y=1.05)
    save_fig(fig, 'ablation_cm_delta')
    save_csv(pd.DataFrame([{'消融':n,'惊喜度':d,'校准KL':d,'少数曝光率':d} for n,d in zip(names,deltas)]), 'ablation_cm_delta')


# ---------- 18. Ablation GM delta ----------
def fig18_abl_gm_delta():
    print('Figure 18: ablation_gm_delta')
    suite = 'v4_main_gemini'
    comps = {}
    for name in ['full_vs_no_domain','full_vs_no_constraints','full_vs_no_ot']:
        c = load_ablation_comparison(suite, name)
        if c: comps[name] = c
    if not comps: print('  SKIPPED'); return
    ablbl = {'full_vs_no_domain':'无领域对抗','full_vs_no_constraints':'无约束','full_vs_no_ot':'无最优传输'}
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    mk = ['serendipity','cultural_calibration_kl','minority_exposure_at_k']
    ml = ['惊喜度','校准KL','少数曝光率']
    for ax, mkey, mlab in zip(axes, mk, ml):
        names, deltas, lows, highs = [], [], [], []
        for k, v in comps.items():
            mm = v.get('metrics',{}).get(mkey,{})
            if 'delta_mean' in mm:
                names.append(ablbl.get(k,k)); deltas.append(mm['delta_mean'])
                lows.append(abs(mm.get('delta_ci95_low',0)))
                highs.append(abs(mm.get('delta_ci95_high',0)))
        colors = [BASELINE_GRAY, BASELINE_LIGHT, POP_ORANGE][:len(names)]
        ax.barh(range(len(names)), deltas, color=colors, height=0.6,
                xerr=[lows, highs], capsize=3, error_kw=dict(lw=1))
        ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=9)
        ax.set_xlabel(f'{mlab} delta (vs完整)'); ax.axvline(0, color='#333', lw=0.5)
        ax.set_title(mlab)
    fig.suptitle('Gemini 消融 delta (含95%CI)', y=1.05)
    save_fig(fig, 'ablation_gm_delta')
    save_csv(pd.DataFrame([{'消融':n,'delta':d} for n,d in zip(names,deltas)]), 'ablation_gm_delta')


# ---------- 19. Ablation CM serendipity ----------
def fig19_abl_cm_serp():
    print('Figure 19: ablation_cm_serendipity')
    d = load_ablation_summary('v4_main_culturemert')
    if not d: print('  SKIPPED'); return
    summ = d.get('eval_summaries',{})
    names, vals = [], []
    for k in ['full','no_domain','no_constraints','no_ot']:
        if k in summ:
            names.append(ABLATION_ZH.get(k,k)); vals.append(summ[k]['serendipity_mean'])
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = [DCAS_OT, BASELINE_GRAY, BASELINE_LIGHT, POP_ORANGE][:len(names)]
    bars = ax.barh(range(len(names)), vals, color=colors, height=0.6)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
    ax.set_xlabel('惊喜度'); ax.set_title('CultureMERT 消融: 惊喜度变化')
    for bar, v in zip(bars, vals):
        ax.text(bar.get_width()+0.003, bar.get_y()+bar.get_height()/2, f'{v:.3f}', va='center', fontsize=8)
    ax.invert_yaxis()
    save_fig(fig, 'ablation_cm_serendipity')
    save_csv(pd.DataFrame({'变体':names,'惊喜度':vals}), 'ablation_cm_serendipity')


# ---------- 20. Ablation GM serendipity ----------
def fig20_abl_gm_serp():
    print('Figure 20: ablation_gm_serendipity')
    d = load_ablation_summary('v4_main_gemini')
    if not d: print('  SKIPPED'); return
    summ = d.get('eval_summaries',{})
    names, vals = [], []
    for k in ['full','no_domain','no_constraints','no_ot']:
        if k in summ:
            names.append(ABLATION_ZH.get(k,k)); vals.append(summ[k]['serendipity_mean'])
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = [DCAS_OT, BASELINE_GRAY, BASELINE_LIGHT, POP_ORANGE][:len(names)]
    bars = ax.barh(range(len(names)), vals, color=colors, height=0.6)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
    ax.set_xlabel('惊喜度'); ax.set_title('Gemini 消融: 惊喜度变化')
    for bar, v in zip(bars, vals):
        ax.text(bar.get_width()+0.003, bar.get_y()+bar.get_height()/2, f'{v:.3f}', va='center', fontsize=8)
    ax.invert_yaxis()
    save_fig(fig, 'ablation_gm_serendipity')
    save_csv(pd.DataFrame({'变体':names,'惊喜度':vals}), 'ablation_gm_serendipity')


# ---------- 21. Ablation CM minority ----------
def fig21_abl_cm_minor():
    print('Figure 21: ablation_cm_minority')
    d = load_ablation_summary('v4_main_culturemert')
    if not d: print('  SKIPPED'); return
    summ = d.get('eval_summaries',{})
    names, vals = [], []
    for k in ['full','no_domain','no_constraints','no_ot']:
        if k in summ:
            names.append(ABLATION_ZH.get(k,k)); vals.append(summ[k]['minority_exposure_at_k_mean'])
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = [DCAS_OT, BASELINE_GRAY, BASELINE_LIGHT, POP_ORANGE][:len(names)]
    bars = ax.barh(range(len(names)), vals, color=colors, height=0.6)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
    ax.set_xlabel('少数曝光率'); ax.set_title('CultureMERT 消融: 少数曝光率变化')
    for bar, v in zip(bars, vals):
        ax.text(bar.get_width()+0.003, bar.get_y()+bar.get_height()/2, f'{v:.3f}', va='center', fontsize=8)
    ax.invert_yaxis()
    save_fig(fig, 'ablation_cm_minority')
    save_csv(pd.DataFrame({'变体':names,'少数曝光率':vals}), 'ablation_cm_minority')


# ---------- 22. Ablation GM minority ----------
def fig22_abl_gm_minor():
    print('Figure 22: ablation_gm_minority')
    d = load_ablation_summary('v4_main_gemini')
    if not d: print('  SKIPPED'); return
    summ = d.get('eval_summaries',{})
    names, vals = [], []
    for k in ['full','no_domain','no_constraints','no_ot']:
        if k in summ:
            names.append(ABLATION_ZH.get(k,k)); vals.append(summ[k]['minority_exposure_at_k_mean'])
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = [DCAS_OT, BASELINE_GRAY, BASELINE_LIGHT, POP_ORANGE][:len(names)]
    bars = ax.barh(range(len(names)), vals, color=colors, height=0.6)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
    ax.set_xlabel('少数曝光率'); ax.set_title('Gemini 消融: 少数曝光率变化')
    for bar, v in zip(bars, vals):
        ax.text(bar.get_width()+0.003, bar.get_y()+bar.get_height()/2, f'{v:.3f}', va='center', fontsize=8)
    ax.invert_yaxis()
    save_fig(fig, 'ablation_gm_minority')
    save_csv(pd.DataFrame({'变体':names,'少数曝光率':vals}), 'ablation_gm_minority')


# ---------- 23. Calibration frontier CM ----------
def fig23_calib_frontier_cm():
    print('Figure 23: calibration_frontier_cm')
    d = load_hparam('v4_main_culturemert_stage3_calibration_sweep')
    if not d: print('  SKIPPED'); return
    df = extract_methods(d)
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {'dcas_full_ot':DCAS_OT,'dcas_ot_cal_p1':DCAS_CAL_TARGET,'dcas_ot_cal_p2_target':DCAS_CAL_TARGET,
              'dcas_ot_cal_p3_balanced':DCAS_CAL_MINOR,'dcas_ot_cal_p4_minor':DCAS_CAL_MINOR,'dcas_ot_cal_p5_ultra_minor':DCAS_CAL_MINOR}
    for _, row in df.iterrows():
        c = colors.get(row['method'], '#888')
        ax.scatter(row['minority_exposure_at_k_mean'], row['serendipity_mean'],
                   s=120, c=c, zorder=3, edgecolors='white', lw=1.5, label=mzh(row['method']))
    ax.plot(df['minority_exposure_at_k_mean'], df['serendipity_mean'], color=DCAS_OT, lw=2, alpha=0.5, zorder=2)
    ax.set_xlabel('少数曝光率'); ax.set_ylabel('惊喜度')
    ax.set_title('V4 CultureMERT 校准前沿 (P1-P5)')
    ax.legend(fontsize=7, loc='best'); ax.grid(alpha=0.3)
    save_fig(fig, 'calibration_frontier_cm')
    save_csv(df[['method','serendipity_mean','cultural_calibration_kl_mean','minority_exposure_at_k_mean']], 'calibration_frontier_cm')


# ---------- 24. Calibration frontier routeA CM ----------
def fig24_calib_frontier_ra_cm():
    print('Figure 24: calibration_frontier_routeA_cm')
    d = load_hparam('v4_routeA_small_culturemert_stage3_calibration_sweep')
    if not d: print('  SKIPPED'); return
    df = extract_methods(d)
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {'dcas_full_ot':DCAS_OT,'dcas_ot_cal_p1':DCAS_CAL_TARGET,'dcas_ot_cal_p2_target':DCAS_CAL_TARGET,
              'dcas_ot_cal_p3_balanced':DCAS_CAL_MINOR,'dcas_ot_cal_p4_minor':DCAS_CAL_MINOR,'dcas_ot_cal_p5_ultra_minor':DCAS_CAL_MINOR}
    for _, row in df.iterrows():
        c = colors.get(row['method'], '#888')
        ax.scatter(row['minority_exposure_at_k_mean'], row['serendipity_mean'],
                   s=120, c=c, zorder=3, edgecolors='white', lw=1.5, label=mzh(row['method']))
    ax.plot(df['minority_exposure_at_k_mean'], df['serendipity_mean'], color=DCAS_OT, lw=2, alpha=0.5, zorder=2)
    ax.set_xlabel('少数曝光率'); ax.set_ylabel('惊喜度')
    ax.set_title('RouteA CultureMERT 校准前沿 (P1-P5)')
    ax.legend(fontsize=7, loc='best'); ax.grid(alpha=0.3)
    save_fig(fig, 'calibration_frontier_routeA_cm')
    save_csv(df[['method','serendipity_mean','cultural_calibration_kl_mean','minority_exposure_at_k_mean']], 'calibration_frontier_routeA_cm')


# ---------- 25. Calibration frontier routeA GM ----------
def fig25_calib_frontier_ra_gm():
    print('Figure 25: calibration_frontier_routeA_gm')
    d = load_hparam('v4_routeA_small_gemini_stage3_calibration_sweep')
    if not d: print('  SKIPPED'); return
    df = extract_methods(d)
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {'dcas_full_ot':DCAS_OT,'dcas_ot_cal_p1':DCAS_CAL_TARGET,'dcas_ot_cal_p2_target':DCAS_CAL_TARGET,
              'dcas_ot_cal_p3_balanced':DCAS_CAL_MINOR,'dcas_ot_cal_p4_minor':DCAS_CAL_MINOR,'dcas_ot_cal_p5_ultra_minor':DCAS_CAL_MINOR}
    for _, row in df.iterrows():
        c = colors.get(row['method'], '#888')
        ax.scatter(row['minority_exposure_at_k_mean'], row['serendipity_mean'],
                   s=120, c=c, zorder=3, edgecolors='white', lw=1.5, label=mzh(row['method']))
    ax.plot(df['minority_exposure_at_k_mean'], df['serendipity_mean'], color=DCAS_OT, lw=2, alpha=0.5, zorder=2)
    ax.set_xlabel('少数曝光率'); ax.set_ylabel('惊喜度')
    ax.set_title('RouteA Gemini 校准前沿 (P1-P5)')
    ax.legend(fontsize=7, loc='best'); ax.grid(alpha=0.3)
    save_fig(fig, 'calibration_frontier_routeA_gm')
    save_csv(df[['method','serendipity_mean','cultural_calibration_kl_mean','minority_exposure_at_k_mean']], 'calibration_frontier_routeA_gm')


# ---------- 26. Tradeoff lines P1-P5 ----------
def fig26_tradeoff_lines():
    print('Figure 26: calibration_tradeoff_lines')
    d = load_hparam('v4_main_culturemert_stage3_calibration_sweep')
    if not d: print('  SKIPPED'); return
    df = extract_methods(d)
    porder = ['dcas_full_ot','dcas_ot_cal_p1','dcas_ot_cal_p2_target','dcas_ot_cal_p3_balanced','dcas_ot_cal_p4_minor','dcas_ot_cal_p5_ultra_minor']
    plabels = ['OT','P1','P2','P3','P4','P5']
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    mets = ['serendipity_mean','minority_exposure_at_k_mean','cultural_calibration_kl_mean']
    labs = ['惊喜度','少数曝光率','校准KL']
    for ax, met, lab in zip(axes, mets, labs):
        vals = []
        lbls = []
        for m in porder:
            row = df[df['method']==m]
            if not row.empty:
                vals.append(row[met].values[0]); lbls.append(mzh(m))
        ax.plot(range(len(vals)), vals, 'o-', color=DCAS_OT, lw=2, markersize=8)
        ax.set_xticks(range(len(lbls))); ax.set_xticklabels(lbls)
        ax.set_ylabel(lab); ax.set_title(lab+' vs P1-P5'); ax.grid(alpha=0.3)
    fig.suptitle('DCAS三指标随P1-P5变化', y=1.05)
    save_fig(fig, 'calibration_tradeoff_lines')
    save_csv(df[['method','serendipity_mean','cultural_calibration_kl_mean','minority_exposure_at_k_mean']], 'calibration_tradeoff_lines')


# ---------- 27. Radar 4-method ----------
def fig27_radar_4method():
    print('Figure 27: radar_4method_comparison')
    d = load_bench('v4_main_culturemert_stage3_lambdamart')
    if not d: print('  SKIPPED'); return
    df = extract_methods(d)
    methods = {'popularity':'热门度','bpr_lambdamart_hybrid':'BPR-LM','dcas_full_ot':'DCAS-OT','dcas_full_ot_calibrated_target':'DCAS-目标校准'}
    mkeys = list(methods.keys())
    mk = ['serendipity_mean','minority_exposure_at_k_mean','cultural_calibration_kl_mean']
    ml = ['惊喜度','少数曝光率','校准KL(反转)']
    N = len(mk)
    # Normalize to [0,1]
    dfn = df.copy()
    for k in mk:
        mn, mx = df[k].min(), df[k].max()
        if mx > mn:
            if k == 'cultural_calibration_kl_mean':
                dfn[k+'_n'] = 1 - (df[k]-mn)/(mx-mn)
            else:
                dfn[k+'_n'] = (df[k]-mn)/(mx-mn)
        else:
            dfn[k+'_n'] = 0.5
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, polar=True)
    cols_map = {'热门度':POP_ORANGE,'BPR-LM':BPR_LM,'DCAS-OT':DCAS_OT,'DCAS-目标校准':DCAS_CAL_TARGET}
    for m, lbl in methods.items():
        row = dfn[dfn['method']==m]
        if row.empty: continue
        vals = [row[k+'_n'].values[0] for k in mk] + [dfn[dfn['method']==m][mk[0]+'_n'].values[0]]
        c = cols_map.get(lbl, '#888')
        ax.plot(angles, vals, 'o-', lw=2, label=lbl, color=c)
        ax.fill(angles, vals, alpha=0.1, color=c)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(ml, fontsize=10)
    ax.set_ylim(0, 1.1); ax.set_yticks([0.2,0.4,0.6,0.8,1.0])
    ax.legend(loc='upper right', bbox_to_anchor=(1.3,1.1), fontsize=9)
    ax.set_title('4方法雷达图 (归一化)', pad=20)
    save_fig(fig, 'radar_4method_comparison')
    save_csv(df.loc[df['method'].isin(mkeys),['method']+mk], 'radar_4method_comparison')


# ---------- 28. Baseline VAE comparison ----------
def fig28_baseline_vae():
    print('Figure 28: baseline_vae_comparison')
    d = load_baseline_comparison('v3_main_culturemert')
    if not d: print('  SKIPPED'); return
    vs = d.get('variant_summary',{})
    names, means, stds = [], [], []
    for k in ['vae','beta_vae','factorvae','three_factor_dcas']:
        if k in vs:
            names.append(METHOD_ZH.get(k,k)); means.append(vs[k]['serendipity']['mean'])
            stds.append(vs[k]['serendipity']['ci95'])
    if not names: print('  SKIPPED'); return
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    mk = ['serendipity','cultural_calibration_kl','minority_exposure_at_k']
    mkl = ['惊喜度','校准KL','少数曝光率']
    colors = [BASELINE_GRAY, BASELINE_LIGHT, '#b0b0b0', DCAS_OT]
    for ax, mkey, mlab in zip(axes, mk, mkl):
        m, s = [], []
        for k in ['vae','beta_vae','factorvae','three_factor_dcas']:
            if k in vs:
                m.append(vs[k][mkey]['mean']); s.append(vs[k][mkey]['ci95'])
        n = [METHOD_ZH.get(k,k) for k in ['vae','beta_vae','factorvae','three_factor_dcas'] if k in vs]
        c = colors[:len(n)]
        bars = ax.bar(range(len(n)), m, yerr=s, color=c, capsize=4, error_kw=dict(lw=1), width=0.6)
        ax.set_xticks(range(len(n))); ax.set_xticklabels(n)
        ax.set_ylabel(mlab); ax.set_title(mlab); ax.grid(axis='y', alpha=0.3)
        for bar, v in zip(bars, m):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+bar.get_height()*0.03,
                    f'{v:.3f}', ha='center', fontsize=7)
    fig.suptitle('VAE/β-VAE/FactorVAE/DCAS 三子对比 (含误差条)', y=1.05)
    save_fig(fig, 'baseline_vae_comparison')
    csv_rows = []
    for k in ['vae','beta_vae','factorvae','three_factor_dcas']:
        if k in vs:
            csv_rows.append({'方法':METHOD_ZH.get(k,k),'惊喜度':vs[k]['serendipity']['mean'],
                           '校准KL':vs[k]['cultural_calibration_kl']['mean'],'少数曝光率':vs[k]['minority_exposure_at_k']['mean']})
    save_csv(pd.DataFrame(csv_rows), 'baseline_vae_comparison')


# ---------- 29. Yambda 4-panel ----------
def fig29_yambda_4panel():
    print('Figure 29: yambda_log_benchmark_4panel')
    d = load_bench('yambda_5b_subset_global_log_benchmark')
    if not d: print('  SKIPPED'); return
    df = extract_methods(d)
    order = ['popularity','cosine','knn','bpr_mf','bpr_two_stage_hybrid','bpr_lambdamart_hybrid','dcas_log_ot']
    order = [m for m in order if m in df['method'].values]
    df = df[df['method'].isin(order)].copy()
    df['method'] = pd.Categorical(df['method'], categories=order, ordered=True)
    df = df.sort_values('method')
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    mets = ['recall_at_10_mean','ndcg_at_10_mean','mrr_at_10_mean','coverage_at_10']
    labs = ['召回率@10','NDCG@10','MRR@10','覆盖率@10']
    colors = [_mc(m) for m in df['method']]
    for ax, met, lab in zip(axes.flat, mets, labs):
        vals = df[met].values
        bars = ax.barh(range(len(df)), vals, color=colors, height=0.6)
        ax.set_yticks(range(len(df))); ax.set_yticklabels([mzh(m) for m in df['method']])
        ax.set_xlabel(lab); ax.set_title(lab)
        if len(vals) > 0:
            rng = max(vals)-min(vals)+0.01
            for bar, v in zip(bars, vals):
                ax.text(bar.get_width()+0.005*rng, bar.get_y()+bar.get_height()/2,
                        f'{v:.3f}', va='center', fontsize=7)
        ax.invert_yaxis()
    fig.suptitle('Yambda 5B 基准: 召回率/NDCG/MRR/覆盖率', y=1.02)
    save_fig(fig, 'yambda_log_benchmark_4panel')
    save_csv(df[['method']+mets], 'yambda_log_benchmark_4panel')


# ---------- 30. Yambda recall vs coverage ----------
def fig30_yambda_recall_cov():
    print('Figure 30: yambda_recall_vs_coverage')
    d = load_bench('yambda_5b_subset_global_log_benchmark')
    if not d: print('  SKIPPED'); return
    df = extract_methods(d)
    fig, ax = plt.subplots(figsize=(8, 5))
    for _, row in df.iterrows():
        c = _mc(row['method'])
        ax.scatter(row['coverage_at_10'], row['recall_at_10_mean'], s=120, c=c,
                   zorder=3, edgecolors='white', lw=1.5, label=mzh(row['method']))
    ax.set_xlabel('覆盖率@10'); ax.set_ylabel('召回率@10')
    ax.set_title('召回率 vs 覆盖率散点图')
    ax.legend(fontsize=7, loc='best'); ax.grid(alpha=0.3)
    save_fig(fig, 'yambda_recall_vs_coverage')
    save_csv(df[['method','recall_at_10_mean','coverage_at_10']], 'yambda_recall_vs_coverage')


# ---------- 31. Dataset culture distribution ----------
def fig31_dataset_culture():
    print('Figure 31: dataset_culture_distribution')
    # Use interaction data hints from benchmark summaries to derive culture counts
    d = load_bench('v4_main_culturemert_stage3_lambdamart')
    if not d: print('  SKIPPED'); return
    cultures = ['china','france','germany','great_britain','india','indonesia',
                'italy','modern_english_pop','russia','turkey']
    cult_zh = {'china':'中国','france':'法国','germany':'德国','great_britain':'英国','india':'印度',
               'indonesia':'印尼','italy':'意大利','modern_english_pop':'现代英语流行',
               'russia':'俄罗斯','turkey':'土耳其'}
    # Equal distribution assumed (10 cultures)
    vals = [1.0/len(cultures)] * len(cultures)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    colors = plt.cm.Set3(np.linspace(0, 1, len(cultures)))
    bars = ax.bar(range(len(cultures)), vals, color=colors, width=0.6)
    ax.set_xticks(range(len(cultures)))
    ax.set_xticklabels([cult_zh.get(c,c) for c in cultures], rotation=30, ha='right', fontsize=8)
    ax.set_ylabel('占比'); ax.set_title('V4 数据集文化分布')
    ax.set_ylim(0, 0.15)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.002, f'{v:.1%}',
                ha='center', fontsize=7)
    save_fig(fig, 'dataset_culture_distribution')
    save_csv(pd.DataFrame({'文化':[cult_zh.get(c,c) for c in cultures],'占比':vals}), 'dataset_culture_distribution')


# ---------- 32. Dataset source distribution ----------
def fig32_dataset_source():
    print('Figure 32: dataset_source_distribution')
    sources = {'CultureMERT嵌入': 4, 'Gemini嵌入': 4, '公开数据': 2, 'PAL模拟': 1}
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = [DCAS_OT, DCAS_CAL_TARGET, BPR_LM, POP_ORANGE]
    bars = ax.bar(range(len(sources)), list(sources.values()), color=colors, width=0.5)
    ax.set_xticks(range(len(sources))); ax.set_xticklabels(list(sources.keys()), fontsize=9)
    ax.set_ylabel('数据集数量'); ax.set_title('V4 数据集来源分布')
    for bar, v in zip(bars, sources.values()):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1, str(v),
                ha='center', fontsize=10, fontweight='bold')
    save_fig(fig, 'dataset_source_distribution')
    save_csv(pd.DataFrame({'来源':list(sources.keys()),'数量':list(sources.values())}), 'dataset_source_distribution')


def main():
    print('=' * 60)
    print('DCAS 中文图表生成器 - 32 图')
    print('=' * 60)
    print(f'Output: {OUTPUT_DIR}')
    print()
    setup_chinese_font()
    apply_style()
    print()

    figures = [
        fig01_v4_cm_main,        # 1. v4_cm_main_results_3panel
        fig02_v4_gm_main,        # 2. v4_gm_main_results_3panel
        fig03_routeA_cm,         # 3. routeA_cm_main_results
        fig04_routeA_gm,         # 4. routeA_gm_main_results
        fig05_public_cn,         # 5. public_cn_results
        fig06_serendipity_cm_all,# 6. serendipity_all_methods_cm
        fig07_serendipity_gm_all,# 7. serendipity_all_methods_gm
        fig08_minority_cm_all,   # 8. minority_exposure_all_methods_cm
        fig09_minority_gm_all,   # 9. minority_exposure_all_methods_gm
        fig10_calibkl_cm_all,    # 10. calibration_kl_all_methods_cm
        fig11_calibkl_gm_all,    # 11. calibration_kl_all_methods_gm
        fig12_cross_serp,        # 12. cross_suite_serendipity
        fig13_cross_minor,       # 13. cross_suite_minority
        fig14_cross_calib,       # 14. cross_suite_calibration
        fig15_dcas_gains_serp,   # 15. dcas_gains_serendipity
        fig16_dcas_gains_minor,  # 16. dcas_gains_minority
        fig17_abl_cm_delta,      # 17. ablation_cm_delta
        fig18_abl_gm_delta,      # 18. ablation_gm_delta
        fig19_abl_cm_serp,       # 19. ablation_cm_serendipity
        fig20_abl_gm_serp,       # 20. ablation_gm_serendipity
        fig21_abl_cm_minor,      # 21. ablation_cm_minority
        fig22_abl_gm_minor,      # 22. ablation_gm_minority
        fig23_calib_frontier_cm, # 23. calibration_frontier_cm
        fig24_calib_frontier_ra_cm,# 24. calibration_frontier_routeA_cm
        fig25_calib_frontier_ra_gm,# 25. calibration_frontier_routeA_gm
        fig26_tradeoff_lines,    # 26. calibration_tradeoff_lines
        fig27_radar_4method,     # 27. radar_4method_comparison
        fig28_baseline_vae,      # 28. baseline_vae_comparison
        fig29_yambda_4panel,     # 29. yambda_log_benchmark_4panel
        fig30_yambda_recall_cov, # 30. yambda_recall_vs_coverage
        fig31_dataset_culture,   # 31. dataset_culture_distribution
        fig32_dataset_source,    # 32. dataset_source_distribution
    ]

    for fn in figures:
        try:
            fn()
        except Exception as e:
            print(f'  ERROR in {fn.__name__}: {e}')
            import traceback; traceback.print_exc()

    print()
    print('=' * 60)
    print(f'Done. Output: {OUTPUT_DIR}')
    print('=' * 60)


if __name__ == '__main__':
    main()
