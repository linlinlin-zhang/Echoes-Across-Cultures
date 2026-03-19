from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dcas.data.npz_tracks import load_tracks


@dataclass(frozen=True)
class DatasetBundle:
    name: str
    title: str
    metadata_csv: Path
    interactions_csv: Path
    tracks_npz: Path


@dataclass(frozen=True)
class BenchmarkBundle:
    name: str
    title: str
    summary_json: Path


def _bundle_paths() -> tuple[list[DatasetBundle], list[BenchmarkBundle], Path]:
    datasets = [
        DatasetBundle(
            name="v3_main",
            title="Research Dataset V3",
            metadata_csv=REPO_ROOT / "storage/public/research_dataset_v3/metadata_v3_main_harmonized_mw3.csv",
            interactions_csv=REPO_ROOT / "storage/public/research_dataset_v3/interactions_v3_main_mixed_mw3.csv",
            tracks_npz=REPO_ROOT / "storage/public/research_dataset_v3/tracks_culturemert_v3_main_mw3.npz",
        ),
        DatasetBundle(
            name="routeA_phase2_cn",
            title="Public RouteA Phase2 CN",
            metadata_csv=REPO_ROOT / "storage/public/routeA_phase2_cn/metadata_merged.csv",
            interactions_csv=REPO_ROOT / "storage/public/routeA_phase2_cn/interactions.csv",
            tracks_npz=REPO_ROOT / "storage/public/routeA_phase2_cn/tracks.npz",
        ),
    ]
    benchmarks = [
        BenchmarkBundle(
            name="v3_main_culturemert_stage3_lambdamart",
            title="V3 Main CultureMERT Stage3 LambdaMART",
            summary_json=REPO_ROOT / "reports/benchmarks/v3_main_culturemert_stage3_lambdamart/benchmark_summary.json",
        ),
        BenchmarkBundle(
            name="public_routeA_phase2_cn_lambdamart",
            title="Public RouteA Phase2 CN LambdaMART",
            summary_json=REPO_ROOT / "reports/benchmarks/public_routeA_phase2_cn_lambdamart/benchmark_summary.json",
        ),
    ]
    pal_summary = REPO_ROOT / "reports/routeA_phase3_pal_cn/phase3_pal_summary.json"
    return datasets, benchmarks, pal_summary


def _save_figure(fig: plt.Figure, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _series_barh(
    values: pd.Series,
    title: str,
    xlabel: str,
    out_path: Path,
    color: str = "#2A6F97",
) -> None:
    if values.empty:
        return
    s = values.sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(9, max(3.6, 0.42 * len(s) + 1.2)))
    ax.barh(s.index.astype(str), s.values.astype(float), color=color)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    for idx, value in enumerate(s.values.astype(float).tolist()):
        ax.text(float(value), idx, f" {int(value):,}", va="center", fontsize=8)
    _save_figure(fig, out_path)


def _histogram(
    values: pd.Series,
    title: str,
    xlabel: str,
    out_path: Path,
    bins: int = 20,
    color: str = "#4D908E",
) -> None:
    if values.empty:
        return
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ax.hist(values.astype(float).to_numpy(), bins=int(bins), color=color, edgecolor="white")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    _save_figure(fig, out_path)


def _stacked_bar(
    frame: pd.DataFrame,
    title: str,
    out_path: Path,
    ylabel: str = "count",
) -> None:
    if frame.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 5.2))
    frame.plot(kind="bar", stacked=True, ax=ax, colormap="tab20")
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, title="")
    _save_figure(fig, out_path)


def _load_benchmark_metrics(path: Path, suite_name: str) -> pd.DataFrame:
    obj = json.loads(path.read_text(encoding="utf-8"))
    methods = obj.get("methods", {})
    rows: list[dict[str, Any]] = []
    for method_name, metrics in methods.items():
        if not isinstance(metrics, dict):
            continue
        rows.append(
            {
                "suite": suite_name,
                "method": str(method_name),
                "serendipity": float(metrics.get("serendipity_mean", float("nan"))),
                "cultural_calibration_kl": float(metrics.get("cultural_calibration_kl_mean", float("nan"))),
                "minority_exposure_at_k": float(metrics.get("minority_exposure_at_k_mean", float("nan"))),
                "target_culture_prob_mean": float(metrics.get("target_culture_prob_mean", float("nan"))),
            }
        )
    return pd.DataFrame(rows)


def _benchmark_bar_grid(df: pd.DataFrame, title: str, out_path: Path) -> None:
    if df.empty:
        return
    metric_specs = [
        ("serendipity", True, "#2A9D8F"),
        ("cultural_calibration_kl", False, "#E76F51"),
        ("minority_exposure_at_k", True, "#577590"),
        ("target_culture_prob_mean", True, "#BC6C25"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5))
    for ax, (metric, higher_is_better, color) in zip(axes.flatten(), metric_specs):
        subset = df[["method", metric]].copy().sort_values(metric, ascending=not higher_is_better)
        ax.barh(subset["method"], subset[metric], color=color)
        ax.set_title(metric)
        if higher_is_better:
            ax.invert_yaxis()
        for idx, value in enumerate(subset[metric].tolist()):
            ax.text(float(value), idx, f" {value:.3f}", va="center", fontsize=8)
    fig.suptitle(title)
    _save_figure(fig, out_path)


def _benchmark_frontier(df: pd.DataFrame, title: str, out_path: Path) -> None:
    if df.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8))
    plots = [
        ("minority_exposure_at_k", "target_culture_prob_mean", "Serendipity vs Minority", "#2A9D8F"),
        ("target_culture_prob_mean", "cultural_calibration_kl", "Target Affinity vs Calibration", "#577590"),
    ]
    for ax, (x_metric, y_metric, subtitle, color) in zip(axes, plots):
        ax.scatter(df[x_metric], df[y_metric], s=72, color=color, alpha=0.85)
        for _, row in df.iterrows():
            ax.annotate(
                str(row["method"]),
                (float(row[x_metric]), float(row[y_metric])),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=8,
            )
        ax.set_xlabel(x_metric)
        ax.set_ylabel(y_metric)
        ax.set_title(subtitle)
    fig.suptitle(title)
    _save_figure(fig, out_path)


def _plot_embedding_pca(bundle: DatasetBundle, out_dir: Path) -> dict[str, str]:
    if not bundle.tracks_npz.exists():
        return {}
    tracks = load_tracks(str(bundle.tracks_npz))
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(tracks.embedding.astype(np.float32))
    frame = pd.DataFrame(
        {
            "pc1": coords[:, 0],
            "pc2": coords[:, 1],
            "culture": tracks.culture.astype(str),
        }
    )
    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    cultures = sorted(frame["culture"].unique().tolist())
    cmap = plt.get_cmap("tab10")
    for idx, culture in enumerate(cultures):
        subset = frame[frame["culture"] == culture]
        ax.scatter(
            subset["pc1"],
            subset["pc2"],
            s=18,
            alpha=0.72,
            label=culture,
            color=cmap(idx % 10),
        )
    ax.set_title(f"{bundle.title}: PCA of track embeddings")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, title="culture")
    out_path = out_dir / f"{bundle.name}_embedding_pca.png"
    _save_figure(fig, out_path)
    return {f"{bundle.name}_embedding_pca": str(out_path.resolve())}


def _summarize_dataset(bundle: DatasetBundle, out_dir: Path) -> tuple[dict[str, Any], list[dict[str, str]]]:
    if not bundle.metadata_csv.exists() or not bundle.interactions_csv.exists():
        return {}, []
    metadata = pd.read_csv(bundle.metadata_csv)
    interactions = pd.read_csv(bundle.interactions_csv)

    figures: list[dict[str, str]] = []

    culture_counts = metadata["culture"].astype(str).value_counts().sort_values(ascending=False)
    culture_csv = out_dir / f"{bundle.name}_counts_by_culture.csv"
    culture_counts.rename_axis("culture").reset_index(name="n_tracks").to_csv(culture_csv, index=False)
    culture_fig = out_dir / f"{bundle.name}_counts_by_culture.png"
    _series_barh(culture_counts, f"{bundle.title}: tracks by culture", "tracks", culture_fig, color="#1D3557")
    figures.append({"name": f"{bundle.name}_counts_by_culture", "path": str(culture_fig.resolve())})

    if "source_dataset" in metadata.columns:
        source_counts = metadata["source_dataset"].fillna("unknown").astype(str).value_counts().sort_values(ascending=False)
        source_csv = out_dir / f"{bundle.name}_counts_by_source.csv"
        source_counts.rename_axis("source_dataset").reset_index(name="n_tracks").to_csv(source_csv, index=False)
        source_fig = out_dir / f"{bundle.name}_counts_by_source.png"
        _series_barh(source_counts, f"{bundle.title}: tracks by source", "tracks", source_fig, color="#457B9D")
        figures.append({"name": f"{bundle.name}_counts_by_source", "path": str(source_fig.resolve())})

        culture_source = pd.crosstab(metadata["culture"].astype(str), metadata["source_dataset"].fillna("unknown").astype(str))
        culture_source_fig = out_dir / f"{bundle.name}_culture_by_source.png"
        _stacked_bar(culture_source, f"{bundle.title}: culture/source composition", culture_source_fig)
        figures.append({"name": f"{bundle.name}_culture_by_source", "path": str(culture_source_fig.resolve())})

    label_col = "coarse_label" if "coarse_label" in metadata.columns else ("label" if "label" in metadata.columns else None)
    if label_col:
        label_counts = (
            metadata[label_col]
            .fillna("unknown")
            .astype(str)
            .value_counts()
            .head(12)
            .sort_values(ascending=False)
        )
        label_csv = out_dir / f"{bundle.name}_top_labels.csv"
        label_counts.rename_axis(label_col).reset_index(name="n_tracks").to_csv(label_csv, index=False)
        label_fig = out_dir / f"{bundle.name}_top_labels.png"
        _series_barh(label_counts, f"{bundle.title}: top labels", "tracks", label_fig, color="#A8DADC")
        figures.append({"name": f"{bundle.name}_top_labels", "path": str(label_fig.resolve())})

    by_user = interactions.groupby("user_id").agg(
        n_interactions=("track_id", "size"),
        weight_sum=("weight", "sum"),
    )
    hist_fig = out_dir / f"{bundle.name}_interactions_per_user_hist.png"
    _histogram(by_user["n_interactions"], f"{bundle.title}: interactions per user", "interactions per user", hist_fig, bins=20)
    figures.append({"name": f"{bundle.name}_interactions_per_user_hist", "path": str(hist_fig.resolve())})

    weight_fig = out_dir / f"{bundle.name}_interaction_weight_hist.png"
    _histogram(interactions["weight"], f"{bundle.title}: interaction weight distribution", "interaction weight", weight_fig, bins=24, color="#F4A261")
    figures.append({"name": f"{bundle.name}_interaction_weight_hist", "path": str(weight_fig.resolve())})

    track_culture = metadata[["track_id", "culture"]].copy()
    merged = interactions.merge(track_culture, on="track_id", how="left")
    culture_cov = merged.groupby("user_id")["culture"].nunique().rename("n_cultures")
    cov_fig = out_dir / f"{bundle.name}_culture_coverage_per_user.png"
    _histogram(culture_cov, f"{bundle.title}: culture coverage per user", "unique cultures per user", cov_fig, bins=10, color="#8D99AE")
    figures.append({"name": f"{bundle.name}_culture_coverage_per_user", "path": str(cov_fig.resolve())})

    figures.extend(
        {"name": key, "path": value}
        for key, value in _plot_embedding_pca(bundle=bundle, out_dir=out_dir).items()
    )

    summary = {
        "dataset": bundle.name,
        "title": bundle.title,
        "n_tracks": int(len(metadata)),
        "n_cultures": int(metadata["culture"].nunique()),
        "n_sources": int(metadata["source_dataset"].nunique()) if "source_dataset" in metadata.columns else None,
        "n_users": int(interactions["user_id"].nunique()),
        "n_interactions": int(len(interactions)),
        "mean_interactions_per_user": float(by_user["n_interactions"].mean()),
        "mean_unique_cultures_per_user": float(culture_cov.mean()),
    }
    return summary, figures


def _summarize_benchmarks(bundles: list[BenchmarkBundle], out_dir: Path) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    all_rows: list[pd.DataFrame] = []
    figures: list[dict[str, str]] = []
    for bundle in bundles:
        if not bundle.summary_json.exists():
            continue
        frame = _load_benchmark_metrics(bundle.summary_json, bundle.title)
        if frame.empty:
            continue
        csv_path = out_dir / f"{bundle.name}_metrics.csv"
        frame.to_csv(csv_path, index=False)
        bar_path = out_dir / f"{bundle.name}_metric_grid.png"
        _benchmark_bar_grid(frame, f"{bundle.title}: four-metric comparison", bar_path)
        frontier_path = out_dir / f"{bundle.name}_frontier.png"
        _benchmark_frontier(frame, f"{bundle.title}: frontier views", frontier_path)
        figures.append({"name": f"{bundle.name}_metric_grid", "path": str(bar_path.resolve())})
        figures.append({"name": f"{bundle.name}_frontier", "path": str(frontier_path.resolve())})
        all_rows.append(frame)
    if not all_rows:
        return pd.DataFrame(), figures
    combined = pd.concat(all_rows, ignore_index=True)
    combined.to_csv(out_dir / "benchmark_metrics_combined.csv", index=False)
    return combined, figures


def _pal_task_frame(path: Path, round_name: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            obj = json.loads(raw)
            rows.append(
                {
                    "round": round_name,
                    "culture": str(obj.get("culture", "")),
                    "uncertainty": float(obj.get("uncertainty", float("nan"))),
                }
            )
    return pd.DataFrame(rows)


def _summarize_pal(summary_path: Path, out_dir: Path) -> tuple[dict[str, Any], list[dict[str, str]]]:
    if not summary_path.exists():
        return {}, []
    obj = json.loads(summary_path.read_text(encoding="utf-8"))
    figures: list[dict[str, str]] = []

    rows_df = pd.DataFrame(obj.get("rows", []))
    rounds_df = pd.DataFrame(
        [
            {
                "round": int(row.get("round", 0)),
                "task_count": int(row.get("task_info", {}).get("count", 0)),
                "n_constraints": int(row.get("round_constraints_report", {}).get("n_constraints", 0)),
                "n_positive": int(row.get("round_constraints_report", {}).get("n_positive", 0)),
                "n_negative": int(row.get("round_constraints_report", {}).get("n_negative", 0)),
                "n_merged_constraints": int(row.get("n_merged_constraints", 0)),
            }
            for row in obj.get("rounds", [])
        ]
    )
    if not rows_df.empty:
        rows_csv = out_dir / "pal_rows.csv"
        rows_df.to_csv(rows_csv, index=False)
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
        axes[0].plot(rows_df["tag"], rows_df["serendipity_mean"], marker="o", color="#2A9D8F")
        axes[0].set_title("PAL serendipity trajectory")
        axes[0].set_ylabel("serendipity_mean")
        axes[1].plot(rows_df["tag"], rows_df["cultural_calibration_kl_mean"], marker="o", color="#E76F51")
        axes[1].set_title("PAL calibration trajectory")
        axes[1].set_ylabel("cultural_calibration_kl_mean")
        pal_metric_path = out_dir / "pal_round_metric_trajectory.png"
        _save_figure(fig, pal_metric_path)
        figures.append({"name": "pal_round_metric_trajectory", "path": str(pal_metric_path.resolve())})
    if not rounds_df.empty:
        rounds_df.to_csv(out_dir / "pal_round_constraints.csv", index=False)
        fig, ax = plt.subplots(figsize=(8.8, 4.8))
        ax.bar(rounds_df["round"].astype(str), rounds_df["n_positive"], label="positive", color="#2A9D8F")
        ax.bar(
            rounds_df["round"].astype(str),
            rounds_df["n_negative"],
            bottom=rounds_df["n_positive"],
            label="negative",
            color="#E76F51",
        )
        ax.plot(rounds_df["round"].astype(str), rounds_df["n_merged_constraints"], marker="o", color="#264653", label="merged")
        ax.set_title("PAL constraints by round")
        ax.set_xlabel("round")
        ax.set_ylabel("count")
        ax.legend(frameon=False)
        constraint_path = out_dir / "pal_constraint_flow.png"
        _save_figure(fig, constraint_path)
        figures.append({"name": "pal_constraint_flow", "path": str(constraint_path.resolve())})

    task_frames: list[pd.DataFrame] = []
    for round_row in obj.get("rounds", []):
        task_path_raw = round_row.get("task_info", {}).get("tasks")
        round_idx = int(round_row.get("round", 0))
        if not task_path_raw:
            continue
        task_path = (REPO_ROOT / str(task_path_raw)).resolve()
        if task_path.exists():
            task_frames.append(_pal_task_frame(task_path, round_name=f"round{round_idx}"))
    if task_frames:
        task_df = pd.concat(task_frames, ignore_index=True)
        task_df.to_csv(out_dir / "pal_task_distribution.csv", index=False)
        culture_counts = pd.crosstab(task_df["culture"], task_df["round"])
        task_culture_path = out_dir / "pal_task_culture_distribution.png"
        _stacked_bar(culture_counts, "PAL task distribution by culture", task_culture_path)
        figures.append({"name": "pal_task_culture_distribution", "path": str(task_culture_path.resolve())})

        fig, ax = plt.subplots(figsize=(8.8, 4.6))
        for round_name, subset in task_df.groupby("round"):
            ax.hist(subset["uncertainty"], bins=18, alpha=0.55, label=round_name)
        ax.set_title("PAL uncertainty distribution")
        ax.set_xlabel("uncertainty")
        ax.set_ylabel("count")
        ax.legend(frameon=False)
        uncertainty_path = out_dir / "pal_uncertainty_distribution.png"
        _save_figure(fig, uncertainty_path)
        figures.append({"name": "pal_uncertainty_distribution", "path": str(uncertainty_path.resolve())})

    summary = {
        "n_rounds": int(len(obj.get("rounds", []))),
        "rows_recorded": int(len(obj.get("rows", []))),
        "baseline_serendipity": float(rows_df.iloc[0]["serendipity_mean"]) if not rows_df.empty else None,
        "final_serendipity": float(rows_df.iloc[-1]["serendipity_mean"]) if not rows_df.empty else None,
        "final_merged_constraints": int(rounds_df["n_merged_constraints"].iloc[-1]) if not rounds_df.empty else None,
    }
    return summary, figures


def _write_markdown_summary(
    out_path: Path,
    dataset_summaries: list[dict[str, Any]],
    benchmark_rows: pd.DataFrame,
    pal_summary: dict[str, Any],
    figures: list[dict[str, str]],
) -> None:
    lines = ["# Project Figure Pack", ""]
    if dataset_summaries:
        lines.append("## Dataset snapshots")
        for item in dataset_summaries:
            lines.append(
                "- "
                f"{item['title']}: {item['n_tracks']} tracks, {item['n_cultures']} cultures, "
                f"{item['n_users']} users, {item['n_interactions']} interactions"
            )
        lines.append("")
    if not benchmark_rows.empty:
        lines.append("## Benchmark snapshots")
        for suite_name, suite_df in benchmark_rows.groupby("suite"):
            best_ser = suite_df.sort_values("serendipity", ascending=False).iloc[0]
            best_min = suite_df.sort_values("minority_exposure_at_k", ascending=False).iloc[0]
            lines.append(
                f"- {suite_name}: best serendipity = {best_ser['method']} ({best_ser['serendipity']:.4f}); "
                f"best minority exposure = {best_min['method']} ({best_min['minority_exposure_at_k']:.4f})"
            )
        lines.append("")
    if pal_summary:
        lines.append("## PAL snapshot")
        lines.append(
            "- "
            f"{pal_summary.get('n_rounds', 0)} rounds, "
            f"baseline serendipity {pal_summary.get('baseline_serendipity', float('nan')):.4f}, "
            f"final serendipity {pal_summary.get('final_serendipity', float('nan')):.4f}, "
            f"final merged constraints {pal_summary.get('final_merged_constraints', 0)}"
        )
        lines.append("")
    lines.append("## Figures")
    for item in figures:
        lines.append(f"- {item['name']}: `{item['path']}`")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_project_figures(out_dir: Path) -> dict[str, Any]:
    datasets, benchmarks, pal_summary_path = _bundle_paths()
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_summaries: list[dict[str, Any]] = []
    manifest_figures: list[dict[str, str]] = []

    for bundle in datasets:
        summary, figures = _summarize_dataset(bundle=bundle, out_dir=out_dir)
        if summary:
            dataset_summaries.append(summary)
        manifest_figures.extend(figures)

    benchmark_rows, benchmark_figures = _summarize_benchmarks(bundles=benchmarks, out_dir=out_dir)
    manifest_figures.extend(benchmark_figures)

    pal_summary, pal_figures = _summarize_pal(summary_path=pal_summary_path, out_dir=out_dir)
    manifest_figures.extend(pal_figures)

    summary_md = out_dir / "README.md"
    _write_markdown_summary(
        out_path=summary_md,
        dataset_summaries=dataset_summaries,
        benchmark_rows=benchmark_rows,
        pal_summary=pal_summary,
        figures=manifest_figures,
    )

    manifest = {
        "out_dir": str(out_dir.resolve()),
        "datasets": dataset_summaries,
        "pal": pal_summary,
        "figures": manifest_figures,
        "benchmark_rows": int(len(benchmark_rows)),
        "summary_markdown": str(summary_md.resolve()),
    }
    manifest_path = out_dir / "figure_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest["manifest_json"] = str(manifest_path.resolve())
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate a reusable figure pack for the current project state.")
    ap.add_argument(
        "--out_dir",
        default=str(REPO_ROOT / "reports/figures/project_overview_2026-03-19"),
        help="Output directory for PNG, CSV, and summary artifacts.",
    )
    args = ap.parse_args()
    manifest = generate_project_figures(out_dir=Path(str(args.out_dir)))
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
