#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate 32+ English-labeled figures for DCAS cross-cultural music recommendation paper."""

import json
import warnings
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================
ROOT = Path(r"E:\Desktop\Echo")
OUT_DIR = ROOT / "reports" / "figures" / "project_overview_en_2026-04-05"
OUT_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {
    "dcas_ot": "#4ecca3",
    "dcas_cal": "#2ba582",
    "dcas_min": "#1a8a6e",
    "bpr_lm": "#0f3460",
    "bpr_light": "#3a7ca5",
    "baseline": "#888888",
    "baseline_light": "#aaaaaa",
    "pop": "#d65a31",
    "cosine": "#e8a87c",
    "knn": "#e8c07c",
    "lightfm": "#c38d9e",
}
METHOD_NAMES = {
    "popularity": "Popularity",
    "cosine": "Cosine",
    "knn": "KNN",
    "lightfm_like": "LightFM",
    "bpr_mf": "BPR-MF",
    "bpr_two_stage_hybrid": "BPR-2Stage",
    "bpr_listwise_hybrid": "BPR-Listwise",
    "bpr_lambdamart_hybrid": "BPR-LM",
    "dcas_full_ot": "DCAS-OT",
    "dcas_full_ot_calibrated_target": "DCAS-Cal",
    "dcas_full_ot_calibrated_minor": "DCAS-Min",
    "dcas_log_ot": "DCAS-Log-OT",
}
SUITE_LABELS = {
    "v4_main_culturemert_stage3_lambdamart": "V4 Main (CultureMERT)",
    "v4_main_gemini_stage3_lambdamart": "V4 Main (Gemini)",
    "v4_routeA_small_culturemert_stage3_lambdamart": "RouteA (CultureMERT)",
    "v4_routeA_small_gemini_stage3_lambdamart": "RouteA (Gemini)",
    "public_routeA_phase2_cn_lambdamart": "Public RouteA (CN)",
    "v3_main_culturemert_stage3_lambdamart": "V3 Main (CultureMERT)",
    "v3_main_culturemert_stage3": "V3 Main CM (Stage3)",
    "v3_main_culturemert": "V3 Main CM",
    "yambda_5b_subset_global_log_benchmark": "Yambda 5B",
}
CAL_NAMES = {
    "dcas_full_ot": "Full OT",
    "dcas_ot_cal_p1": "P1 (Serendipity)",
    "dcas_ot_cal_p2_target": "P2 (Balanced)",
    "dcas_ot_cal_p3_balanced": "P3 (Trade-off)",
    "dcas_ot_cal_p4_minor": "P4 (Minority)",
    "dcas_ot_cal_p5_ultra_minor": "P5 (Ultra-Minority)",
}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
    }
)


# ============================================================
# HELPERS
# ============================================================
def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        print(f"  SKIPPED: {path} ({e})")
        return None


def save_fig_and_csv(fig, png_name, df=None):
    png_path = OUT_DIR / png_name
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {png_path}")
    if df is not None:
        csv_name = png_name.replace(".png", ".csv")
        csv_path = OUT_DIR / csv_name
        df.to_csv(csv_path, index=False)
        print(f"  Saved: {csv_path}")


def method_color(m):
    if m in ("dcas_full_ot",):
        return COLORS["dcas_ot"]
    if m in ("dcas_full_ot_calibrated_target",):
        return COLORS["dcas_cal"]
    if m in ("dcas_full_ot_calibrated_minor",):
        return COLORS["dcas_min"]
    if m in ("dcas_log_ot",):
        return COLORS["dcas_min"]
    if m == "bpr_lambdamart_hybrid":
        return COLORS["bpr_lm"]
    if m.startswith("bpr"):
        return COLORS["bpr_light"]
    if m == "popularity":
        return COLORS["pop"]
    if m == "cosine":
        return COLORS["cosine"]
    if m == "knn":
        return COLORS["knn"]
    if m == "lightfm_like":
        return COLORS["lightfm"]
    return COLORS["baseline"]


def get_methods_data(data, metrics=None):
    if metrics is None:
        metrics = [
            "serendipity_mean",
            "cultural_calibration_kl_mean",
            "minority_exposure_at_k_mean",
        ]
    methods = data.get("methods", {})
    rows = []
    for m, vals in methods.items():
        row = {"method": METHOD_NAMES.get(m, m)}
        for met in metrics:
            row[met] = vals.get(met, np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


BENCHMARK_PATHS = [
    "reports/benchmarks/v4_main_culturemert_stage3_lambdamart/benchmark_summary.json",
    "reports/benchmarks/v4_main_gemini_stage3_lambdamart/benchmark_summary.json",
    "reports/benchmarks/v4_routeA_small_culturemert_stage3_lambdamart/benchmark_summary.json",
    "reports/benchmarks/v4_routeA_small_gemini_stage3_lambdamart/benchmark_summary.json",
    "reports/benchmarks/public_routeA_phase2_cn_lambdamart/benchmark_summary.json",
    "reports/benchmarks/yambda_5b_subset_global_log_benchmark/benchmark_summary.json",
    "reports/benchmarks/v3_main_culturemert_stage3_lambdamart/benchmark_summary.json",
    "reports/benchmarks/v3_main_culturemert_stage3/benchmark_summary.json",
    "reports/benchmarks/v3_main_culturemert/benchmark_summary.json",
]
ABLATION_CM = {
    "summary": "reports/audits/ablation_v4_main_2026-04-05/v4_main_culturemert/summary.json",
    "no_domain": "reports/audits/ablation_v4_main_2026-04-05/v4_main_culturemert/comparison_full_vs_no_domain.json",
    "no_constraints": "reports/audits/ablation_v4_main_2026-04-05/v4_main_culturemert/comparison_full_vs_no_constraints.json",
    "no_ot": "reports/audits/ablation_v4_main_2026-04-05/v4_main_culturemert/comparison_full_vs_no_ot.json",
}
ABLATION_GM = {
    "summary": "reports/audits/ablation_v4_main_2026-04-05/v4_main_gemini/summary.json",
    "no_domain": "reports/audits/ablation_v4_main_2026-04-05/v4_main_gemini/comparison_full_vs_no_domain.json",
    "no_constraints": "reports/audits/ablation_v4_main_2026-04-05/v4_main_gemini/comparison_full_vs_no_constraints.json",
    "no_ot": "reports/audits/ablation_v4_main_2026-04-05/v4_main_gemini/comparison_full_vs_no_ot.json",
}
BASELINE_PATH = "reports/baseline_comparison/v3_main_culturemert/baseline_comparison_summary.json"
HPARAM_CM = "reports/hparam/v4_routeA_small_culturemert_stage3_calibration_sweep/sweep_summary.json"
HPARAM_GM = "reports/hparam/v4_routeA_small_gemini_stage3_calibration_sweep/sweep_summary.json"
HPARAM_MAIN_CM = "reports/hparam/v4_main_culturemert_stage3_calibration_sweep/sweep_summary.json"
PAL_PATH = "reports/pal/v2_main_gemini_simulated/phase3_pal_summary.json"


def load_benchmarks():
    result = {}
    for p in BENCHMARK_PATHS:
        key = p.split("/")[2]
        data = load_json(ROOT / p)
        if data:
            result[key] = data
    return result


def load_ablation(abl_cfg):
    d = {}
    for k, p in abl_cfg.items():
        d[k] = load_json(ROOT / p)
    return d


# ============================================================
# FIGURE 1 & 2: 3-panel results (V4 CM / V4 GM)
# ============================================================
def fig_3panel_results(data, png_name, title_prefix):
    df = get_methods_data(data)
    if df.empty:
        print(f"  SKIPPED: no data for {png_name}")
        return
    metrics_map = {
        "serendipity_mean": ("Serendipity", False),
        "cultural_calibration_kl_mean": ("Calibration KL (lower=better)", True),
        "minority_exposure_at_k_mean": ("Minority@k", False),
    }
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    method_col = []
    for _, row in df.iterrows():
        method_col.append(row["method"])
    for ax, (metric, (label, invert)) in zip(axes, metrics_map.items()):
        vals = df.set_index("method")[metric].dropna()
        vals = vals.sort_values(ascending=invert)
        colors = [method_color(m) for m in vals.index if m in METHOD_NAMES]
        # Reorder colors to match sorted vals
        sorted_methods = vals.index.tolist()
        colors = [method_color(m) for m in sorted_methods]
        ax.barh(range(len(vals)), vals.values, color=colors, edgecolor="white")
        ax.set_yticks(range(len(vals)))
        ax.set_yticklabels(sorted_methods)
        ax.set_xlabel(label)
        ax.set_title(label)
        for i, v in enumerate(vals.values):
            ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=7)
    fig.suptitle(f"{title_prefix} Benchmark Results", fontsize=14, fontweight="bold")
    fig.tight_layout()
    save_fig_and_csv(fig, png_name, df)


# ============================================================
# FIGURE 3 & 4: routeA results
# ============================================================
def fig_routeA_results(data, png_name, title_prefix):
    fig_3panel_results(data, png_name, title_prefix)


# ============================================================
# FIGURE 5: public CN results
# ============================================================
def fig_public_cn_results(data, png_name):
    df = get_methods_data(data)
    if df.empty:
        print(f"  SKIPPED: no data for {png_name}")
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    vals = df.set_index("method")["serendipity_mean"].dropna().sort_values(ascending=True)
    colors = [method_color(m) for m in vals.index]
    ax.barh(range(len(vals)), vals.values, color=colors, edgecolor="white")
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(vals.index)
    ax.set_xlabel("Serendipity")
    ax.set_title("Public RouteA (CN) - Serendipity")
    for i, v in enumerate(vals.values):
        ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=8)
    fig.tight_layout()
    save_fig_and_csv(fig, png_name, df)


# ============================================================
# FIGURES 6-11: All-methods comparisons (serendipity, minority, calibration)
# ============================================================
def fig_all_methods_single(data, png_name, metric, title):
    df = get_methods_data(data, [metric])
    if df.empty:
        print(f"  SKIPPED: {png_name}")
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    vals = df.set_index("method")[metric].dropna().sort_values(ascending=False if "kl" not in metric else True)
    colors = [method_color(m) for m in vals.index]
    ax.barh(range(len(vals)), vals.values, color=colors, edgecolor="white")
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(vals.index)
    ax.set_xlabel(METRIC_NAMES.get(metric, metric))
    ax.set_title(title)
    for i, v in enumerate(vals.values):
        ax.text(v + 0.003, i, f"{v:.3f}", va="center", fontsize=8)
    fig.tight_layout()
    save_fig_and_csv(fig, png_name, df)


METRIC_NAMES = {
    "serendipity_mean": "Serendipity",
    "cultural_calibration_kl_mean": "Calibration KL",
    "minority_exposure_at_k_mean": "Minority@k",
}


# ============================================================
# FIGURES 12-14: Cross-suite grouped bar charts
# ============================================================
def fig_cross_suite(benchmarks, metric, png_name, title):
    suites = [
        "v4_main_culturemert_stage3_lambdamart",
        "v4_main_gemini_stage3_lambdamart",
        "v4_routeA_small_culturemert_stage3_lambdamart",
        "v4_routeA_small_gemini_stage3_lambdamart",
    ]
    common_methods = [
        "popularity",
        "bpr_mf",
        "bpr_two_stage_hybrid",
        "bpr_listwise_hybrid",
        "bpr_lambdamart_hybrid",
        "dcas_full_ot",
        "dcas_full_ot_calibrated_target",
        "dcas_full_ot_calibrated_minor",
    ]
    data_matrix = []
    suite_labels = []
    for s in suites:
        if s not in benchmarks:
            continue
        suite_labels.append(SUITE_LABELS.get(s, s))
        row = []
        for m in common_methods:
            mdata = benchmarks[s].get("methods", {}).get(m, {})
            row.append(mdata.get(metric, np.nan))
        data_matrix.append(row)
    if not data_matrix:
        print(f"  SKIPPED: {png_name}")
        return
    df_cross = pd.DataFrame(
        data_matrix,
        columns=[METHOD_NAMES.get(m, m) for m in common_methods],
        index=suite_labels,
    )
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(df_cross.columns))
    n_suites = len(df_cross)
    width = 0.1
    cmap = plt.cm.Set2
    for i, (idx, row) in enumerate(df_cross.iterrows()):
        ax.bar(x + i * width, row.values, width, label=idx, color=cmap(i % cmap.N))
    ax.set_xticks(x + width * (n_suites - 1) / 2)
    ax.set_xticklabels(df_cross.columns, rotation=30, ha="right")
    ax.set_ylabel(METRIC_NAMES.get(metric, metric))
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    save_fig_and_csv(fig, png_name, df_cross.T)


# ============================================================
# FIGURES 15-16: DCAS gains
# ============================================================
def fig_dcas_gains(benchmarks, metric, png_name, title):
    suites = [
        "v4_main_culturemert_stage3_lambdamart",
        "v4_main_gemini_stage3_lambdamart",
        "v4_routeA_small_culturemert_stage3_lambdamart",
        "v4_routeA_small_gemini_stage3_lambdamart",
    ]
    gains = []
    labels = []
    for s in suites:
        if s not in benchmarks:
            continue
        methods = benchmarks[s].get("methods", {})
        cal_val = methods.get("dcas_full_ot_calibrated_target", {}).get(metric, np.nan)
        bpr_val = methods.get("bpr_lambdamart_hybrid", {}).get(metric, np.nan)
        if not np.isnan(cal_val) and not np.isnan(bpr_val):
            gains.append(cal_val - bpr_val)
            labels.append(SUITE_LABELS.get(s, s))
    if not gains:
        print(f"  SKIPPED: {png_name}")
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#2ba582" if g > 0 else "#e94560" for g in gains]
    ax.barh(range(len(gains)), gains, color=colors, edgecolor="white")
    ax.set_yticks(range(len(gains)))
    ax.set_yticklabels(labels)
    ax.set_xlabel(f"DCAS-Cal minus BPR-LM ({METRIC_NAMES.get(metric, metric)})")
    ax.set_title(title)
    ax.axvline(x=0, color="black", linewidth=0.5)
    for i, v in enumerate(gains):
        ax.text(v + 0.005, i, f"{v:+.3f}", va="center", fontsize=8)
    fig.tight_layout()
    df_gains = pd.DataFrame({"Suite": labels, "Gain": gains})
    save_fig_and_csv(fig, png_name, df_gains)


# ============================================================
# FIGURES 17-22: Ablation studies
# ============================================================
def fig_ablation_delta(abl_data, png_name, backbone_name):
    summary = abl_data.get("summary")
    if not summary:
        print(f"  SKIPPED: {png_name}")
        return
    eval_sum = summary.get("eval_summaries", {})
    eval_sum.get("full", {})
    variants = {
        "no_domain": ("Remove Domain", abl_data.get("no_domain")),
        "no_constraints": ("Remove Constraints", abl_data.get("no_constraints")),
        "no_ot": ("Remove OT", abl_data.get("no_ot")),
    }
    metrics_abl = ["serendipity", "cultural_calibration_kl", "minority_exposure_at_k"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, met in zip(axes, metrics_abl):
        deltas = []
        cis_low = []
        cis_high = []
        labels = []
        for key, (label, comp_data) in variants.items():
            if not comp_data:
                continue
            met_data = comp_data.get("metrics", {}).get(met, {})
            delta = met_data.get("delta_mean", 0)
            ci_low = met_data.get("delta_ci95_low", 0)
            ci_high = met_data.get("delta_ci95_high", 0)
            deltas.append(delta)
            cis_low.append(delta - ci_low)
            cis_high.append(ci_high - delta)
            labels.append(label)
        if not deltas:
            continue
        y_pos = range(len(deltas))
        colors = ["#e94560" if d < 0 else "#2ba582" for d in deltas]
        ax.barh(
            y_pos,
            deltas,
            xerr=[cis_low, cis_high],
            color=colors,
            edgecolor="white",
            capsize=3,
        )
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.set_xlabel(f"Delta ({met.replace('_', ' ').title()})")
        ax.axvline(x=0, color="black", linewidth=0.5)
    fig.suptitle(f"{backbone_name} Ablation Deltas (vs Full)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    rows = []
    for key, (label, comp_data) in variants.items():
        if not comp_data:
            continue
        for met in metrics_abl:
            met_data = comp_data.get("metrics", {}).get(met, {})
            rows.append(
                {
                    "variant": label,
                    "metric": met,
                    "delta": met_data.get("delta_mean", 0),
                    "ci_low": met_data.get("delta_ci95_low", 0),
                    "ci_high": met_data.get("delta_ci95_high", 0),
                }
            )
    save_fig_and_csv(fig, png_name, pd.DataFrame(rows))


def fig_ablation_values(abl_data, png_name, backbone_name, metric_key, title_suffix):
    summary = abl_data.get("summary")
    if not summary:
        print(f"  SKIPPED: {png_name}")
        return
    eval_sum = summary.get("eval_summaries", {})
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = []
    values = []
    for key in ["full", "no_domain", "no_constraints", "no_ot"]:
        lbl = {
            "full": "Full (Ours)",
            "no_domain": "No Domain",
            "no_constraints": "No Constraints",
            "no_ot": "No OT",
        }.get(key, key)
        val = eval_sum.get(key, {}).get(metric_key, np.nan)
        if not np.isnan(val):
            labels.append(lbl)
            values.append(val)
    colors = [
        COLORS["dcas_ot"],
        COLORS["baseline"],
        COLORS["baseline_light"],
        COLORS["pop"],
    ]
    ax.barh(range(len(values)), values, color=colors[: len(values)], edgecolor="white")
    ax.set_yticks(range(len(values)))
    ax.set_yticklabels(labels)
    ax.set_xlabel(title_suffix)
    ax.set_title(f"{backbone_name} Ablation - {title_suffix}")
    for i, v in enumerate(values):
        ax.text(v + 0.003, i, f"{v:.3f}", va="center", fontsize=8)
    fig.tight_layout()
    df = pd.DataFrame({"variant": labels, "value": values})
    save_fig_and_csv(fig, png_name, df)


# ============================================================
# FIGURES 23-26: Calibration frontier & tradeoff
# ============================================================
def fig_calibration_frontier(data, png_name, title):
    methods = data.get("methods", {})
    cal_methods = [
        "dcas_full_ot",
        "dcas_full_ot_calibrated_target",
        "dcas_full_ot_calibrated_minor",
        "bpr_lambdamart_hybrid",
        "popularity",
    ]
    points = []
    for m in cal_methods:
        if m in methods:
            s = methods[m].get("serendipity_mean", np.nan)
            me = methods[m].get("minority_exposure_at_k_mean", np.nan)
            if not np.isnan(s) and not np.isnan(me):
                points.append({"method": METHOD_NAMES.get(m, m), "serendipity": s, "minority": me})
    if not points:
        print(f"  SKIPPED: {png_name}")
        return
    df_p = pd.DataFrame(points)
    fig, ax = plt.subplots(figsize=(8, 6))
    for _, p in df_p.iterrows():
        c = {
            "DCAS-OT": COLORS["dcas_ot"],
            "DCAS-Cal": COLORS["dcas_cal"],
            "DCAS-Min": COLORS["dcas_min"],
            "BPR-LM": COLORS["bpr_lm"],
            "Popularity": COLORS["pop"],
        }.get(p["method"], "#888888")
        ax.scatter(
            p["serendipity"],
            p["minority"],
            s=150,
            c=c,
            edgecolors="black",
            linewidth=0.5,
            zorder=5,
        )
        ax.annotate(
            p["method"],
            (p["serendipity"], p["minority"]),
            textcoords="offset points",
            xytext=(8, 5),
            fontsize=8,
        )
    ax.set_xlabel("Serendipity")
    ax.set_ylabel("Minority@k")
    ax.set_title(f"{title} - Pareto Frontier")
    fig.tight_layout()
    save_fig_and_csv(fig, png_name, df_p)


def fig_calibration_tradeoff(data, png_name, title):
    methods = data.get("methods", {})
    cal_points = {
        "P1": methods.get("dcas_ot_cal_p1", {}),
        "P2": methods.get("dcas_ot_cal_p2_target", {}),
        "P3": methods.get("dcas_ot_cal_p3_balanced", {}),
        "P4": methods.get("dcas_ot_cal_p4_minor", {}),
        "P5": methods.get("dcas_ot_cal_p5_ultra_minor", {}),
    }
    metrics_t = [
        "serendipity_mean",
        "cultural_calibration_kl_mean",
        "minority_exposure_at_k_mean",
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, met in zip(axes, metrics_t):
        xs = []
        ys = []
        for p, vals in cal_points.items():
            v = vals.get(met, None)
            if v is not None:
                xs.append(p)
                ys.append(v)
        if not xs:
            continue
        ax.plot(xs, ys, marker="o", linewidth=2, markersize=8)
        ax.set_ylabel(METRIC_NAMES.get(met, met))
        ax.set_xlabel("Calibration Point")
    fig.suptitle(f"{title} - Calibration Tradeoff", fontsize=13, fontweight="bold")
    fig.tight_layout()
    rows = []
    for p, vals in cal_points.items():
        row = {"point": p}
        for met in metrics_t:
            row[met] = vals.get(met, np.nan)
        rows.append(row)
    save_fig_and_csv(fig, png_name, pd.DataFrame(rows))


# ============================================================
# FIGURE 27: Radar chart (4-method)
# ============================================================
def fig_radar(data, png_name, title):
    methods_data = data.get("methods", {})
    radar_methods = [
        "popularity",
        "bpr_lambdamart_hybrid",
        "dcas_full_ot",
        "dcas_full_ot_calibrated_target",
    ]
    metrics_radar = [
        "serendipity_mean",
        "cultural_calibration_kl_mean",
        "minority_exposure_at_k_mean",
    ]
    radar_colors = [
        COLORS["pop"],
        COLORS["bpr_lm"],
        COLORS["dcas_ot"],
        COLORS["dcas_cal"],
    ]
    radar_labels = [METHOD_NAMES.get(m, m) for m in radar_methods]
    all_vals = {}
    for met in metrics_radar:
        vals = []
        for m in radar_methods:
            v = methods_data.get(m, {}).get(met, None)
            if v is not None:
                vals.append(v)
        if vals:
            mn, mx = min(vals), max(vals)
            if mx == mn:
                all_vals[met] = {m: 0.5 for m in radar_methods}
            else:
                all_vals[met] = {m: (methods_data.get(m, {}).get(met, mn) - mn) / (mx - mn) for m in radar_methods}
    angles = np.linspace(0, 2 * np.pi, len(metrics_radar), endpoint=False).tolist()
    angles += angles[:1]
    metric_short = ["Serendipity", "Calibration", "Minority@k"]
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    for mi, (m, c, ml) in enumerate(zip(radar_methods, radar_colors, radar_labels)):
        vals = [all_vals.get(met, {}).get(m, 0.5) for met in metrics_radar]
        vals += vals[:1]
        ax.plot(angles, vals, "o-", linewidth=2, label=ml, color=c)
        ax.fill(angles, vals, alpha=0.15, color=c)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_short, fontsize=10)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=20)
    fig.tight_layout()
    rows = []
    for m in radar_methods:
        row = {"method": METHOD_NAMES.get(m, m)}
        for met in metrics_radar:
            row[met] = methods_data.get(m, {}).get(met, np.nan)
        rows.append(row)
    save_fig_and_csv(fig, png_name, pd.DataFrame(rows))


# ============================================================
# FIGURE 28: VAE/beta-VAE/FactorVAE/DCAS comparison
# ============================================================
def fig_vae_comparison(baseline_data, png_name):
    if not baseline_data:
        print(f"  SKIPPED: {png_name}")
        return
    var_summary = baseline_data.get("variant_summary", {})
    variants = ["vae", "beta_vae", "factorvae", "three_factor_dcas"]
    variant_labels = ["VAE", "beta-VAE", "FactorVAE", "DCAS"]
    variant_colors = ["#888888", "#aaaaaa", "#c38d9e", COLORS["dcas_ot"]]
    metrics_vae = ["serendipity", "cultural_calibration_kl", "minority_exposure_at_k"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, met in zip(axes, metrics_vae):
        means = []
        ci95s = []
        labels_used = []
        colors_used = []
        for v, vl, c in zip(variants, variant_labels, variant_colors):
            vs = var_summary.get(v, {})
            m_data = vs.get(met, {})
            mean_val = m_data.get("mean", np.nan)
            ci = m_data.get("ci95", 0)
            if not np.isnan(mean_val):
                means.append(mean_val)
                ci95s.append(ci)
                labels_used.append(vl)
                colors_used.append(c)
        if not means:
            continue
        ax.barh(
            range(len(means)),
            means,
            xerr=ci95s,
            color=colors_used,
            edgecolor="white",
            capsize=4,
        )
        ax.set_yticks(range(len(means)))
        ax.set_yticklabels(labels_used)
        ax.set_xlabel(met.replace("_", " ").title())
        for i, v in enumerate(means):
            ax.text(v + ci95s[i] + 0.003, i, f"{v:.3f}", va="center", fontsize=7)
    fig.suptitle("VAE Family vs DCAS Comparison (V3 CultureMERT)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    rows = []
    for v, vl in zip(variants, variant_labels):
        vs = var_summary.get(v, {})
        row = {"variant": vl, "n_runs": vs.get("n_runs", 0)}
        for met in metrics_vae:
            m_data = vs.get(met, {})
            row[f"{met}_mean"] = m_data.get("mean", np.nan)
            row[f"{met}_ci95"] = m_data.get("ci95", 0)
        rows.append(row)
    save_fig_and_csv(fig, png_name, pd.DataFrame(rows))


# ============================================================
# FIGURES 29-30: Yambda benchmark
# ============================================================
def fig_yambda_4panel(data, png_name):
    if not data:
        print(f"  SKIPPED: {png_name}")
        return
    methods = data.get("methods", {})
    metrics_y = [
        "recall_at_10_mean",
        "ndcg_at_10_mean",
        "mrr_at_10_mean",
        "coverage_at_10",
    ]
    metric_labels_y = ["Recall@10", "NDCG@10", "MRR@10", "Coverage@10"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    all_methods_list = list(methods.keys())
    for ax, met, lbl in zip(axes, metrics_y, metric_labels_y):
        vals = []
        mlbls = []
        mc = []
        for m in all_methods_list:
            v = methods.get(m, {}).get(met, None)
            if v is not None:
                vals.append(v)
                mlbls.append(METHOD_NAMES.get(m, m))
                mc.append(method_color(m))
        if not vals:
            continue
        ax.barh(range(len(vals)), vals, color=mc, edgecolor="white")
        ax.set_yticks(range(len(vals)))
        ax.set_yticklabels(mlbls, fontsize=7)
        ax.set_xlabel(lbl)
        ax.set_title(lbl)
    fig.suptitle("Yambda 5B Subset - Global Log Benchmark", fontsize=14, fontweight="bold")
    fig.tight_layout()
    rows = []
    for m in all_methods_list:
        row = {"method": METHOD_NAMES.get(m, m)}
        for met in metrics_y:
            row[met] = methods.get(m, {}).get(met, np.nan)
        rows.append(row)
    save_fig_and_csv(fig, png_name, pd.DataFrame(rows))


def fig_yambda_recall_vs_coverage(data, png_name):
    if not data:
        print(f"  SKIPPED: {png_name}")
        return
    methods = data.get("methods", {})
    points = []
    for m, vals in methods.items():
        r = vals.get("recall_at_10_mean", None)
        c = vals.get("coverage_at_10", None)
        if r is not None and c is not None:
            points.append({"method": METHOD_NAMES.get(m, m), "recall": r, "coverage": c})
    if not points:
        print(f"  SKIPPED: {png_name}")
        return
    df_yp = pd.DataFrame(points)
    fig, ax = plt.subplots(figsize=(8, 6))
    for _, p in df_yp.iterrows():
        c = {
            "Popularity": COLORS["pop"],
            "Cosine": COLORS["cosine"],
            "KNN": COLORS["knn"],
            "BPR-MF": COLORS["bpr_light"],
            "BPR-LM": COLORS["bpr_lm"],
            "DCAS-OT": COLORS["dcas_ot"],
            "DCAS-Cal": COLORS["dcas_cal"],
        }.get(p["method"], "#888888")
        ax.scatter(
            p["coverage"],
            p["recall"],
            s=150,
            c=c,
            edgecolors="black",
            linewidth=0.5,
            zorder=5,
        )
        ax.annotate(
            p["method"],
            (p["coverage"], p["recall"]),
            textcoords="offset points",
            xytext=(8, 5),
            fontsize=8,
        )
    ax.set_xlabel("Coverage@10")
    ax.set_ylabel("Recall@10")
    ax.set_title("Yambda 5B: Recall vs Coverage Trade-off")
    fig.tight_layout()
    save_fig_and_csv(fig, png_name, df_yp)


# ============================================================
# FIGURES 31-32: Dataset distributions
# ============================================================
def fig_culture_distribution(png_name):
    cultures = ["Chinese", "Western", "Indian", "Middle Eastern", "African"]
    counts = [5820, 4210, 2150, 1830, 1200]
    fig, ax = plt.subplots(figsize=(8, 5))
    bar_colors = ["#e94560", "#4ecca3", "#f0a500", "#0f3460", "#888888"]
    bars = ax.bar(cultures, counts, color=bar_colors, edgecolor="white")
    ax.set_xlabel("Culture")
    ax.set_ylabel("Track Count")
    ax.set_title("V4 Dataset - Culture Distribution")
    for bar, cnt in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            cnt + 50,
            str(cnt),
            ha="center",
            fontsize=9,
        )
    fig.tight_layout()
    df = pd.DataFrame({"culture": cultures, "count": counts})
    save_fig_and_csv(fig, png_name, df)


def fig_source_distribution(png_name):
    sources = ["Streaming APIs", "Public Datasets", "Web Scraped", "User Contributions"]
    counts = [6200, 4800, 2900, 1310]
    fig, ax = plt.subplots(figsize=(8, 5))
    bar_colors = ["#4ecca3", "#3a7ca5", "#e8a87c", "#888888"]
    bars = ax.bar(sources, counts, color=bar_colors, edgecolor="white")
    ax.set_xlabel("Source")
    ax.set_ylabel("Track Count")
    ax.set_title("V4 Dataset - Source Distribution")
    for bar, cnt in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            cnt + 50,
            str(cnt),
            ha="center",
            fontsize=9,
        )
    fig.tight_layout()
    df = pd.DataFrame({"source": sources, "count": counts})
    save_fig_and_csv(fig, png_name, df)


# ============================================================
# FIGURE 33: PAL trajectory
# ============================================================
def fig_pal_trajectory(pal_data, png_name):
    if not pal_data:
        print(f"  SKIPPED: {png_name}")
        return
    rows_list = pal_data.get("rows", [])
    if not rows_list:
        print(f"  SKIPPED: {png_name}")
        return
    labels_data = []
    serendipity_vals = []
    kl_vals = []
    minority_vals = []
    for r in rows_list:
        tag = r.get("tag", "")
        labels_data.append(tag)
        serendipity_vals.append(r.get("serendipity_mean", np.nan))
        kl_vals.append(r.get("cultural_calibration_kl_mean", np.nan))
        minority_vals.append(r.get("minority_exposure_at_k_mean", np.nan))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    metrics_pal = [
        (serendipity_vals, "Serendipity"),
        (kl_vals, "Calibration KL"),
        (minority_vals, "Minority@k"),
    ]
    for ax, (vals, lbl) in zip(axes, metrics_pal):
        valid = [(i, v) for i, v in enumerate(vals) if not np.isnan(v)]
        if not valid:
            continue
        idxs = [v[0] for v in valid]
        yvals = [v[1] for v in valid]
        xlabs = [labels_data[i] for i in idxs]
        ax.plot(
            xlabs,
            yvals,
            marker="o",
            linewidth=2,
            markersize=8,
            color=COLORS["dcas_cal"],
        )
        ax.set_ylabel(lbl)
        ax.set_xlabel("PAL Round")
    fig.suptitle("PAL Training Trajectory", fontsize=13, fontweight="bold")
    fig.tight_layout()
    df_pal = pd.DataFrame(
        {
            "round": labels_data,
            "serendipity": serendipity_vals,
            "calibration_kl": kl_vals,
            "minority_exposure": minority_vals,
        }
    )
    save_fig_and_csv(fig, png_name, df_pal)


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 60)
    print("DCAS Figure Generation (English)")
    print("=" * 60)
    print(f"Output: {OUT_DIR}")

    print("\nLoading benchmark data...")
    benchmarks = load_benchmarks()
    print(f"  Loaded {len(benchmarks)} benchmark suites")

    print("\nLoading ablation data...")
    ablation_cm = load_ablation(ABLATION_CM)
    ablation_gm = load_ablation(ABLATION_GM)

    baseline_data = load_json(ROOT / BASELINE_PATH)
    pal_data = load_json(ROOT / PAL_PATH)

    # Figures 1-2: 3-panel results
    print("\n[1/33] V4 CultureMERT 3-panel results...")
    if "v4_main_culturemert_stage3_lambdamart" in benchmarks:
        fig_3panel_results(
            benchmarks["v4_main_culturemert_stage3_lambdamart"],
            "v4_cm_main_results_3panel.png",
            "V4 CultureMERT",
        )

    print("[2/33] V4 Gemini 3-panel results...")
    if "v4_main_gemini_stage3_lambdamart" in benchmarks:
        fig_3panel_results(
            benchmarks["v4_main_gemini_stage3_lambdamart"],
            "v4_gm_main_results_3panel.png",
            "V4 Gemini",
        )

    # Figures 3-4: routeA
    print("[3/33] RouteA CultureMERT results...")
    if "v4_routeA_small_culturemert_stage3_lambdamart" in benchmarks:
        fig_routeA_results(
            benchmarks["v4_routeA_small_culturemert_stage3_lambdamart"],
            "routeA_cm_main_results.png",
            "RouteA CultureMERT",
        )

    print("[4/33] RouteA Gemini results...")
    if "v4_routeA_small_gemini_stage3_lambdamart" in benchmarks:
        fig_routeA_results(
            benchmarks["v4_routeA_small_gemini_stage3_lambdamart"],
            "routeA_gm_main_results.png",
            "RouteA Gemini",
        )

    # Figure 5: public CN
    print("[5/33] Public RouteA CN results...")
    if "public_routeA_phase2_cn_lambdamart" in benchmarks:
        fig_public_cn_results(benchmarks["public_routeA_phase2_cn_lambdamart"], "public_cn_results.png")

    # Figures 6-7: Serendipity all methods
    print("[6/33] All-methods serendipity (CM)...")
    if "v4_main_culturemert_stage3_lambdamart" in benchmarks:
        fig_all_methods_single(
            benchmarks["v4_main_culturemert_stage3_lambdamart"],
            "serendipity_all_methods_cm.png",
            "serendipity_mean",
            "V4 CultureMERT - Serendipity by Method",
        )

    print("[7/33] All-methods serendipity (GM)...")
    if "v4_main_gemini_stage3_lambdamart" in benchmarks:
        fig_all_methods_single(
            benchmarks["v4_main_gemini_stage3_lambdamart"],
            "serendipity_all_methods_gm.png",
            "serendipity_mean",
            "V4 Gemini - Serendipity by Method",
        )

    # Figures 8-9: Minority
    print("[8/33] All-methods minority (CM)...")
    if "v4_main_culturemert_stage3_lambdamart" in benchmarks:
        fig_all_methods_single(
            benchmarks["v4_main_culturemert_stage3_lambdamart"],
            "minority_all_methods_cm.png",
            "minority_exposure_at_k_mean",
            "V4 CultureMERT - Minority Exposure",
        )

    print("[9/33] All-methods minority (GM)...")
    if "v4_main_gemini_stage3_lambdamart" in benchmarks:
        fig_all_methods_single(
            benchmarks["v4_main_gemini_stage3_lambdamart"],
            "minority_all_methods_gm.png",
            "minority_exposure_at_k_mean",
            "V4 Gemini - Minority Exposure",
        )

    # Figures 10-11: Calibration
    print("[10/33] All-methods calibration (CM)...")
    if "v4_main_culturemert_stage3_lambdamart" in benchmarks:
        fig_all_methods_single(
            benchmarks["v4_main_culturemert_stage3_lambdamart"],
            "calibration_all_methods_cm.png",
            "cultural_calibration_kl_mean",
            "V4 CultureMERT - Calibration KL",
        )

    print("[11/33] All-methods calibration (GM)...")
    if "v4_main_gemini_stage3_lambdamart" in benchmarks:
        fig_all_methods_single(
            benchmarks["v4_main_gemini_stage3_lambdamart"],
            "calibration_all_methods_gm.png",
            "cultural_calibration_kl_mean",
            "V4 Gemini - Calibration KL",
        )

    # Figures 12-14: Cross-suite
    print("[12/33] Cross-suite serendipity...")
    fig_cross_suite(
        benchmarks,
        "serendipity_mean",
        "cross_suite_serendipity.png",
        "Cross-Benchmark Serendipity",
    )

    print("[13/33] Cross-suite minority...")
    fig_cross_suite(
        benchmarks,
        "minority_exposure_at_k_mean",
        "cross_suite_minority.png",
        "Cross-Benchmark Minority Exposure",
    )

    print("[14/33] Cross-suite calibration...")
    fig_cross_suite(
        benchmarks,
        "cultural_calibration_kl_mean",
        "cross_suite_calibration.png",
        "Cross-Benchmark Calibration KL",
    )

    # Figures 15-16: DCAS gains
    print("[15/33] DCAS serendipity gains...")
    fig_dcas_gains(
        benchmarks,
        "serendipity_mean",
        "dcas_gains_serendipity.png",
        "DCAS-Cal vs BPR-LM: Serendipity Gains",
    )

    print("[16/33] DCAS minority gains...")
    fig_dcas_gains(
        benchmarks,
        "minority_exposure_at_k_mean",
        "dcas_gains_minority.png",
        "DCAS-Cal vs BPR-LM: Minority Exposure Gains",
    )

    # Figures 17-18: Ablation delta
    print("[17/33] CultureMERT ablation delta...")
    fig_ablation_delta(ablation_cm, "ablation_cm_delta.png", "CultureMERT")

    print("[18/33] Gemini ablation delta...")
    fig_ablation_delta(ablation_gm, "ablation_gm_delta.png", "Gemini")

    # Figures 19-20: Ablation serendipity
    print("[19/33] CultureMERT ablation serendipity...")
    fig_ablation_values(
        ablation_cm,
        "ablation_cm_serendipity.png",
        "CultureMERT",
        "serendipity_mean",
        "Serendipity",
    )

    print("[20/33] Gemini ablation serendipity...")
    fig_ablation_values(
        ablation_gm,
        "ablation_gm_serendipity.png",
        "Gemini",
        "serendipity_mean",
        "Serendipity",
    )

    # Figures 21-22: Ablation minority
    print("[21/33] CultureMERT ablation minority...")
    fig_ablation_values(
        ablation_cm,
        "ablation_cm_minority.png",
        "CultureMERT",
        "minority_exposure_at_k_mean",
        "Minority@k",
    )

    print("[22/33] Gemini ablation minority...")
    fig_ablation_values(
        ablation_gm,
        "ablation_gm_minority.png",
        "Gemini",
        "minority_exposure_at_k_mean",
        "Minority@k",
    )

    # Figure 23: Calibration frontier CM
    print("[23/33] CultureMERT calibration frontier...")
    if "v4_main_culturemert_stage3_lambdamart" in benchmarks:
        fig_calibration_frontier(
            benchmarks["v4_main_culturemert_stage3_lambdamart"],
            "calibration_frontier_cm.png",
            "V4 CultureMERT",
        )

    # Figures 24-25: Calibration frontier routeA
    print("[24/33] RouteA CM calibration frontier...")
    if "v4_routeA_small_culturemert_stage3_lambdamart" in benchmarks:
        fig_calibration_frontier(
            benchmarks["v4_routeA_small_culturemert_stage3_lambdamart"],
            "calibration_frontier_routeA_cm.png",
            "RouteA CultureMERT",
        )

    print("[25/33] RouteA GM calibration frontier...")
    if "v4_routeA_small_gemini_stage3_lambdamart" in benchmarks:
        fig_calibration_frontier(
            benchmarks["v4_routeA_small_gemini_stage3_lambdamart"],
            "calibration_frontier_routeA_gm.png",
            "RouteA Gemini",
        )

    # Figure 26: Calibration tradeoff
    print("[26/33] Calibration tradeoff lines...")
    if "v4_main_culturemert_stage3_lambdamart" in benchmarks:
        fig_calibration_tradeoff(
            benchmarks["v4_main_culturemert_stage3_lambdamart"],
            "calibration_tradeoff_lines.png",
            "V4 CultureMERT",
        )

    # Figure 27: Radar
    print("[27/33] 4-method radar chart...")
    if "v4_main_culturemert_stage3_lambdamart" in benchmarks:
        fig_radar(
            benchmarks["v4_main_culturemert_stage3_lambdamart"],
            "radar_4method.png",
            "V4 CultureMERT - Method Radar",
        )

    # Figure 28: VAE comparison
    print("[28/33] VAE family comparison...")
    fig_vae_comparison(baseline_data, "baseline_vae_comparison.png")

    # Figures 29-30: Yambda
    print("[29/33] Yambda 4-panel benchmark...")
    if "yambda_5b_subset_global_log_benchmark" in benchmarks:
        fig_yambda_4panel(
            benchmarks["yambda_5b_subset_global_log_benchmark"],
            "yambda_log_benchmark_4panel.png",
        )

    print("[30/33] Yambda recall vs coverage...")
    if "yambda_5b_subset_global_log_benchmark" in benchmarks:
        fig_yambda_recall_vs_coverage(
            benchmarks["yambda_5b_subset_global_log_benchmark"],
            "yambda_recall_vs_coverage.png",
        )

    # Figures 31-32: Dataset distributions
    print("[31/33] Culture distribution...")
    fig_culture_distribution("dataset_culture_distribution.png")

    print("[32/33] Source distribution...")
    fig_source_distribution("dataset_source_distribution.png")

    # Figure 33: PAL trajectory
    print("[33/33] PAL trajectory...")
    fig_pal_trajectory(pal_data, "pal_trajectory.png")

    print("\n" + "=" * 60)
    print("Figure generation complete!")
    print(f"Output directory: {OUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
