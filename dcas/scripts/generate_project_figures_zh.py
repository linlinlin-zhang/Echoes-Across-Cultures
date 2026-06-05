from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch
from sklearn.decomposition import PCA

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dcas.data.npz_tracks import load_tracks


def _configure_fonts() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "WenQuanYi Zen Hei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _series_barh(values: pd.Series, title: str, xlabel: str, path: Path, color: str) -> None:
    if values.empty:
        return
    s = values.sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(9.2, max(3.8, 0.42 * len(s) + 1.3)))
    ax.barh(s.index.astype(str), s.values.astype(float), color=color)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    for idx, value in enumerate(s.values.astype(float).tolist()):
        ax.text(float(value), idx, f" {value:,.0f}", va="center", fontsize=8)
    _save(fig, path)


def _hist(
    values: pd.Series,
    title: str,
    xlabel: str,
    ylabel: str,
    path: Path,
    color: str,
    bins: int = 20,
) -> None:
    if values.empty:
        return
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.hist(values.astype(float).to_numpy(), bins=bins, color=color, edgecolor="white")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    _save(fig, path)


def _stacked(frame: pd.DataFrame, title: str, ylabel: str, legend_title: str, path: Path) -> None:
    if frame.empty:
        return
    fig, ax = plt.subplots(figsize=(10.4, 5.5))
    frame.plot(kind="bar", stacked=True, ax=ax, colormap="tab20")
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, title=legend_title)
    _save(fig, path)


def _heatmap(
    frame: pd.DataFrame,
    title: str,
    xlabel: str,
    ylabel: str,
    path: Path,
    cmap: str = "Blues",
) -> None:
    if frame.empty:
        return
    fig, ax = plt.subplots(
        figsize=(
            max(6.8, 0.55 * len(frame.columns) + 2.0),
            max(4.5, 0.24 * len(frame.index) + 2.0),
        )
    )
    arr = frame.to_numpy(dtype=float)
    im = ax.imshow(arr, cmap=cmap, aspect="auto")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(np.arange(len(frame.columns)))
    ax.set_xticklabels(frame.columns.astype(str).tolist(), rotation=45, ha="right")
    ax.set_yticks(np.arange(len(frame.index)))
    ax.set_yticklabels(frame.index.astype(str).tolist())
    if arr.size <= 500:
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                ax.text(j, i, f"{arr[i, j]:.0f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    _save(fig, path)


def _dataset_bundles() -> list[dict[str, Any]]:
    return [
        {
            "name": "v3_main",
            "title": "Research Dataset V3 主数据集",
            "metadata": REPO_ROOT / "storage/public/research_dataset_v3/metadata_v3_main_harmonized_mw3.csv",
            "interactions": REPO_ROOT / "storage/public/research_dataset_v3/interactions_v3_main_mixed_mw3.csv",
            "tracks": REPO_ROOT / "storage/public/research_dataset_v3/tracks_culturemert_v3_main_mw3.npz",
        },
        {
            "name": "routeA_phase2_cn",
            "title": "Public RouteA Phase2 CN 公共多文化数据线",
            "metadata": REPO_ROOT / "storage/public/routeA_phase2_cn/metadata_merged.csv",
            "interactions": REPO_ROOT / "storage/public/routeA_phase2_cn/interactions.csv",
            "tracks": REPO_ROOT / "storage/public/routeA_phase2_cn/tracks.npz",
        },
        {
            "name": "yambda_5b_subset",
            "title": "Yambda-5B 官方子集",
            "metadata": REPO_ROOT / "storage/public/yambda_5b_subset/metadata.csv",
            "interactions": REPO_ROOT / "storage/public/yambda_5b_subset/interactions.csv",
            "tracks": REPO_ROOT / "storage/public/yambda_5b_subset/tracks.npz",
        },
    ]


def _benchmark_bundles() -> list[dict[str, Any]]:
    return [
        {
            "name": "v3_main_culturemert_stage3_lambdamart",
            "title": "V3 Main CultureMERT Stage3 LambdaMART",
            "summary": REPO_ROOT / "reports/benchmarks/v3_main_culturemert_stage3_lambdamart/benchmark_summary.json",
        },
        {
            "name": "public_routeA_phase2_cn_lambdamart",
            "title": "Public RouteA Phase2 CN LambdaMART",
            "summary": REPO_ROOT / "reports/benchmarks/public_routeA_phase2_cn_lambdamart/benchmark_summary.json",
        },
    ]


def _load_cross_metrics(path: Path, suite: str) -> pd.DataFrame:
    obj = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for method, metrics in obj.get("methods", {}).items():
        rows.append(
            {
                "suite": suite,
                "method": method,
                "serendipity": float(metrics.get("serendipity_mean", float("nan"))),
                "kl": float(metrics.get("cultural_calibration_kl_mean", float("nan"))),
                "minority": float(metrics.get("minority_exposure_at_k_mean", float("nan"))),
                "target": float(metrics.get("target_culture_prob_mean", float("nan"))),
            }
        )
    return pd.DataFrame(rows)


def _load_log_metrics(path: Path, suite: str) -> pd.DataFrame:
    obj = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for method, metrics in obj.get("methods", {}).items():
        rows.append(
            {
                "suite": suite,
                "method": method,
                "recall_at_10": float(metrics.get("recall_at_10_mean", float("nan"))),
                "recall_at_20": float(metrics.get("recall_at_20_mean", float("nan"))),
                "ndcg_at_20": float(metrics.get("ndcg_at_20_mean", float("nan"))),
                "mrr_at_20": float(metrics.get("mrr_at_20_mean", float("nan"))),
                "coverage_at_20": float(metrics.get("coverage_at_20", float("nan"))),
            }
        )
    return pd.DataFrame(rows)


def _plot_embedding_pca(title: str, tracks_path: Path, path: Path) -> None:
    if not tracks_path.exists():
        return
    tracks = load_tracks(str(tracks_path))
    coords = PCA(n_components=2, random_state=42).fit_transform(tracks.embedding.astype(np.float32))
    frame = pd.DataFrame(
        {
            "pc1": coords[:, 0],
            "pc2": coords[:, 1],
            "culture": tracks.culture.astype(str),
        }
    )
    fig, ax = plt.subplots(figsize=(8.8, 6.4))
    cmap = plt.get_cmap("tab10")
    for idx, culture in enumerate(sorted(frame["culture"].unique().tolist())):
        subset = frame[frame["culture"] == culture]
        ax.scatter(
            subset["pc1"],
            subset["pc2"],
            s=18,
            alpha=0.72,
            color=cmap(idx % 10),
            label=culture,
        )
    ax.set_title(f"{title}：曲目嵌入 PCA")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, title="文化")
    _save(fig, path)


def _plot_dataset(bundle: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    if not bundle["metadata"].exists() or not bundle["interactions"].exists():
        return {}
    md = pd.read_csv(bundle["metadata"])
    it = pd.read_csv(bundle["interactions"])
    title = str(bundle["title"])
    name = str(bundle["name"])

    _series_barh(
        md["culture"].fillna("unknown").astype(str).value_counts(),
        f"{title}：各文化曲目数",
        "曲目数",
        out_dir / f"{name}_counts_by_culture.png",
        "#1D3557",
    )
    if "source_dataset" in md.columns:
        _series_barh(
            md["source_dataset"].fillna("unknown").astype(str).value_counts(),
            f"{title}：各来源曲目数",
            "曲目数",
            out_dir / f"{name}_counts_by_source.png",
            "#457B9D",
        )
        _stacked(
            pd.crosstab(
                md["culture"].astype(str),
                md["source_dataset"].fillna("unknown").astype(str),
            ),
            f"{title}：文化与来源构成",
            "曲目数",
            "来源",
            out_dir / f"{name}_culture_by_source.png",
        )
    _hist(
        it["weight"],
        f"{title}：交互权重分布",
        "交互权重",
        "交互条数",
        out_dir / f"{name}_interaction_weight_hist.png",
        "#F4A261",
        24,
    )
    by_user = it.groupby("user_id").size().rename("n")
    _hist(
        by_user,
        f"{title}：每位用户交互数分布",
        "每位用户交互数",
        "用户数",
        out_dir / f"{name}_interactions_per_user_hist.png",
        "#4D908E",
        20,
    )
    merged = it.merge(md[["track_id", "culture"]], on="track_id", how="left")
    culture_cov = merged.groupby("user_id")["culture"].nunique()
    _hist(
        culture_cov,
        f"{title}：每位用户覆盖文化数",
        "覆盖文化数",
        "用户数",
        out_dir / f"{name}_culture_coverage_per_user.png",
        "#8D99AE",
        10,
    )
    _series_barh(
        merged["culture"].fillna("unknown").astype(str).value_counts(),
        f"{title}：各文化交互量",
        "交互条数",
        out_dir / f"{name}_interaction_counts_by_culture.png",
        "#6D597A",
    )

    user_culture = merged.groupby(["user_id", "culture"]).size().unstack(fill_value=0)
    if not user_culture.empty:
        top_users = (
            user_culture.assign(total=user_culture.sum(axis=1))
            .sort_values("total", ascending=False)
            .drop(columns=["total"])
            .head(40)
        )
        _heatmap(
            top_users,
            f"{title}：用户-文化交互热力图（前 40 位用户）",
            "文化",
            "用户",
            out_dir / f"{name}_user_culture_heatmap_top40.png",
            cmap="YlOrRd",
        )

    if "duration_sec" in md.columns:
        _hist(
            md["duration_sec"].dropna(),
            f"{title}：音频时长分布",
            "时长（秒）",
            "曲目数",
            out_dir / f"{name}_duration_hist.png",
            "#43AA8B",
            24,
        )

    if "played_ratio_pct" in it.columns:
        _hist(
            it["played_ratio_pct"].dropna(),
            f"{title}：played_ratio_pct 分布",
            "played_ratio_pct",
            "交互条数",
            out_dir / f"{name}_played_ratio_hist.png",
            "#F8961E",
            20,
        )
    if "track_length_seconds" in it.columns:
        _hist(
            it["track_length_seconds"].dropna(),
            f"{title}：日志中的曲目时长分布",
            "时长（秒）",
            "交互条数",
            out_dir / f"{name}_track_length_seconds_hist.png",
            "#277DA1",
            24,
        )
    if "is_organic" in it.columns:
        organic = (
            it["is_organic"]
            .fillna(-1)
            .astype(int)
            .map({1: "organic", 0: "recommendation", -1: "unknown"})
            .value_counts()
        )
        _series_barh(
            organic,
            f"{title}：organic 标记分布",
            "交互条数",
            out_dir / f"{name}_organic_flag_counts.png",
            "#F3722C",
        )

    _plot_embedding_pca(title, bundle["tracks"], out_dir / f"{name}_embedding_pca.png")

    return {
        "title": title,
        "n_tracks": int(len(md)),
        "n_cultures": int(md["culture"].nunique()) if "culture" in md.columns else 0,
        "n_users": int(it["user_id"].nunique()) if "user_id" in it.columns else 0,
        "n_interactions": int(len(it)),
    }


def _plot_cross_suite(bundle: dict[str, Any], out_dir: Path) -> pd.DataFrame:
    if not bundle["summary"].exists():
        return pd.DataFrame()
    df = _load_cross_metrics(bundle["summary"], bundle["title"])
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.8))
    specs = [
        ("serendipity", "Serendipity", False, "#2A9D8F"),
        ("kl", "文化校准 KL", True, "#E76F51"),
        ("minority", "少数文化曝光", False, "#577590"),
        ("target", "目标文化概率", False, "#BC6C25"),
    ]
    for ax, (col, label, ascending, color) in zip(axes.flatten(), specs):
        sub = df[["method", col]].sort_values(col, ascending=ascending)
        ax.barh(sub["method"], sub[col], color=color)
        ax.set_title(label)
        ax.set_xlabel("数值")
        if not ascending:
            ax.invert_yaxis()
        for idx, value in enumerate(sub[col].tolist()):
            ax.text(float(value), idx, f" {value:.3f}", va="center", fontsize=8)
    fig.suptitle(f"{bundle['title']}：四指标对比")
    fig.subplots_adjust(top=0.90)
    _save(fig, out_dir / f"{bundle['name']}_metric_grid.png")

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8))
    axes[0].scatter(df["minority"], df["serendipity"], s=76, color="#2A9D8F")
    for _, row in df.iterrows():
        axes[0].annotate(
            str(row["method"]),
            (float(row["minority"]), float(row["serendipity"])),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=8,
        )
    axes[0].set_title("Serendipity 与少数文化曝光")
    axes[0].set_xlabel("少数文化曝光")
    axes[0].set_ylabel("Serendipity")
    axes[1].scatter(df["target"], df["kl"], s=76, color="#577590")
    for _, row in df.iterrows():
        axes[1].annotate(
            str(row["method"]),
            (float(row["target"]), float(row["kl"])),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=8,
        )
    axes[1].set_title("目标文化概率 与文化校准 KL")
    axes[1].set_xlabel("目标文化概率")
    axes[1].set_ylabel("文化校准 KL")
    fig.suptitle(f"{bundle['title']}：前沿视图")
    fig.subplots_adjust(top=0.87)
    _save(fig, out_dir / f"{bundle['name']}_frontier.png")

    ranks = pd.DataFrame(
        {
            "Serendipity": df["serendipity"].rank(method="min", ascending=False),
            "文化校准 KL": df["kl"].rank(method="min", ascending=True),
            "少数文化曝光": df["minority"].rank(method="min", ascending=False),
            "目标文化概率": df["target"].rank(method="min", ascending=False),
        },
        index=df["method"].astype(str),
    )
    _heatmap(
        ranks,
        f"{bundle['title']}：方法名次热力图",
        "指标",
        "方法",
        out_dir / f"{bundle['name']}_rank_heatmap.png",
        cmap="YlGnBu_r",
    )
    return df


def _plot_log_suite(out_dir: Path) -> pd.DataFrame:
    path = REPO_ROOT / "reports/benchmarks/yambda_5b_subset_global_log_benchmark/benchmark_summary.json"
    if not path.exists():
        return pd.DataFrame()
    df = _load_log_metrics(path, "Yambda-5B 子集日志排序基准")
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 8.4))
    specs = [
        ("recall_at_10", "Recall@10", "#2A9D8F"),
        ("recall_at_20", "Recall@20", "#577590"),
        ("ndcg_at_20", "NDCG@20", "#BC6C25"),
        ("mrr_at_20", "MRR@20", "#E76F51"),
    ]
    for ax, (col, label, color) in zip(axes.flatten(), specs):
        sub = df[["method", col]].sort_values(col, ascending=True)
        ax.barh(sub["method"], sub[col], color=color)
        ax.set_title(label)
        ax.set_xlabel("数值")
        for idx, value in enumerate(sub[col].tolist()):
            ax.text(float(value), idx, f" {value:.3f}", va="center", fontsize=8)
    fig.suptitle("Yambda-5B 子集日志排序基准：排序指标对比")
    fig.subplots_adjust(top=0.90)
    _save(fig, out_dir / "yambda_5b_subset_global_log_benchmark_metric_grid.png")

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    ax.scatter(df["coverage_at_20"], df["recall_at_20"], s=80, color="#43AA8B")
    for _, row in df.iterrows():
        ax.annotate(
            str(row["method"]),
            (float(row["coverage_at_20"]), float(row["recall_at_20"])),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=8,
        )
    ax.set_title("Yambda-5B 子集日志排序基准：Coverage@20 与 Recall@20")
    ax.set_xlabel("Coverage@20")
    ax.set_ylabel("Recall@20")
    _save(fig, out_dir / "yambda_5b_subset_global_log_benchmark_coverage_vs_recall.png")
    return df


def _plot_pal(out_dir: Path) -> None:
    path = REPO_ROOT / "reports/routeA_phase3_pal_cn/phase3_pal_summary.json"
    if not path.exists():
        return
    obj = json.loads(path.read_text(encoding="utf-8"))
    rows = pd.DataFrame(obj.get("rows", []))
    rounds = pd.DataFrame(
        [
            {
                "round": int(row.get("round", 0)),
                "tasks": int(row.get("task_info", {}).get("count", 0)),
                "positive": int(row.get("round_constraints_report", {}).get("n_positive", 0)),
                "negative": int(row.get("round_constraints_report", {}).get("n_negative", 0)),
                "merged": int(row.get("n_merged_constraints", 0)),
            }
            for row in obj.get("rounds", [])
        ]
    )
    if not rows.empty:
        fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4))
        axes[0].plot(rows["tag"], rows["serendipity_mean"], marker="o", color="#2A9D8F")
        axes[0].set_title("PAL 的 Serendipity 轨迹")
        axes[0].set_ylabel("Serendipity")
        axes[1].plot(
            rows["tag"],
            rows["cultural_calibration_kl_mean"],
            marker="o",
            color="#E76F51",
        )
        axes[1].set_title("PAL 的文化校准 KL 轨迹")
        axes[1].set_ylabel("文化校准 KL")
        _save(fig, out_dir / "pal_round_metric_trajectory_zh.png")
    if not rounds.empty:
        fig, ax = plt.subplots(figsize=(8.8, 4.8))
        x = rounds["round"].astype(str)
        ax.bar(x, rounds["positive"], color="#2A9D8F", label="正约束")
        ax.bar(
            x,
            rounds["negative"],
            bottom=rounds["positive"],
            color="#E76F51",
            label="负约束",
        )
        ax.plot(x, rounds["merged"], marker="o", color="#264653", label="累计合并约束")
        ax.set_title("PAL 约束流转")
        ax.set_xlabel("轮次")
        ax.set_ylabel("数量")
        ax.legend(frameon=False)
        _save(fig, out_dir / "pal_constraint_flow_zh.png")


def _plot_overview(
    dataset_summaries: list[dict[str, Any]],
    cross_rows: list[pd.DataFrame],
    log_df: pd.DataFrame,
    out_dir: Path,
) -> None:
    if dataset_summaries:
        frame = pd.DataFrame(dataset_summaries)
        fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.6))
        for ax, col, title, color in zip(
            axes,
            ["n_tracks", "n_users", "n_interactions"],
            ["曲目数", "用户数", "交互数"],
            ["#1D3557", "#457B9D", "#E76F51"],
        ):
            ax.bar(frame["title"], frame[col].astype(float), color=color)
            ax.set_title(title)
            ax.tick_params(axis="x", rotation=25)
        fig.suptitle("数据集规模总览")
        fig.subplots_adjust(top=0.84)
        _save(fig, out_dir / "dataset_scale_overview.png")

    fig, ax = plt.subplots(figsize=(12.5, 3.8))
    ax.axis("off")
    table = ax.table(
        cellText=[
            ["数据来源", "公共音频数据 + 自建文化编排", "Yambda-5B 官方日志子集"],
            ["用户交互", "合成或弱监督交互", "真实平台日志交互"],
            ["核心目标", "跨文化探索与校准", "标准排序准确率"],
            [
                "主要指标",
                "Serendipity / KL / target / minority",
                "Recall@K / NDCG@K / MRR@K",
            ],
            ["推荐语义", "有明确文化目标", "target_culture 退化为 global"],
        ],
        colLabels=["维度", "跨文化主线", "公开日志补充线"],
        cellLoc="left",
        colLoc="left",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    ax.set_title("两类评测协议对照")
    _save(fig, out_dir / "evaluation_protocol_comparison.png")

    fig, ax = plt.subplots(figsize=(13.2, 4.2))
    ax.axis("off")
    boxes = [
        (
            0.04,
            0.35,
            0.16,
            0.32,
            "公共音频与元数据\nResearch Dataset V3 / RouteA / Yambda",
        ),
        (0.24, 0.35, 0.16, 0.32, "嵌入构建与 tracks.npz\nCultureMERT / Gemini"),
        (0.44, 0.35, 0.16, 0.32, "交互层\n合成交互 / 公开日志 / PAL 约束"),
        (0.64, 0.35, 0.16, 0.32, "模型层\nBPR / LambdaMART / DCAS"),
        (0.84, 0.35, 0.12, 0.32, "结果层\n图表 / 表格 / 论文"),
    ]
    colors = ["#DCEAF7", "#EAF4E2", "#FDEBD0", "#F5D7D7", "#E8DAEF"]
    for idx, (x, y, w, h, text) in enumerate(boxes):
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02",
            facecolor=colors[idx],
            edgecolor="#355070",
            linewidth=1.2,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2.0, y + h / 2.0, text, ha="center", va="center", fontsize=10)
    for idx in range(len(boxes) - 1):
        x1 = boxes[idx][0] + boxes[idx][2]
        x2 = boxes[idx + 1][0]
        ax.annotate(
            "",
            xy=(x2 - 0.01, 0.51),
            xytext=(x1 + 0.01, 0.51),
            arrowprops=dict(arrowstyle="->", lw=1.5, color="#355070"),
        )
    ax.set_title("项目全流程总览")
    _save(fig, out_dir / "project_pipeline_overview.png")

    if cross_rows:
        cross_df = pd.concat(cross_rows, ignore_index=True)
        best_rows = []
        for suite, suite_df in cross_df.groupby("suite"):
            best_rows.append(
                {
                    "suite": suite,
                    "best_serendipity": float(suite_df["serendipity"].max()),
                    "best_minority": float(suite_df["minority"].max()),
                    "best_target": float(suite_df["target"].max()),
                }
            )
        best_df = pd.DataFrame(best_rows)
        fig, ax = plt.subplots(figsize=(9.8, 4.8))
        x = np.arange(len(best_df))
        width = 0.25
        ax.bar(
            x - width,
            best_df["best_serendipity"],
            width=width,
            color="#2A9D8F",
            label="最佳 Serendipity",
        )
        ax.bar(
            x,
            best_df["best_minority"],
            width=width,
            color="#577590",
            label="最佳少数文化曝光",
        )
        ax.bar(
            x + width,
            best_df["best_target"],
            width=width,
            color="#BC6C25",
            label="最佳目标文化概率",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(best_df["suite"].astype(str).tolist(), rotation=20, ha="right")
        ax.set_ylabel("数值")
        ax.set_title("跨实验线最佳值对比")
        ax.legend(frameon=False)
        _save(fig, out_dir / "cross_suite_best_values.png")

    if not log_df.empty:
        fig, ax = plt.subplots(figsize=(8.8, 4.8))
        ax.barh(log_df["method"], log_df["recall_at_20"], color="#577590")
        ax.set_title("Yambda-5B 子集日志排序基准：Recall@20")
        ax.set_xlabel("Recall@20")
        _save(fig, out_dir / "yambda_recall20_focus.png")


def generate(out_dir: Path) -> dict[str, Any]:
    _configure_fonts()
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_summaries = []
    for bundle in _dataset_bundles():
        summary = _plot_dataset(bundle, out_dir)
        if summary:
            dataset_summaries.append(summary)

    cross_rows = []
    for bundle in _benchmark_bundles():
        df = _plot_cross_suite(bundle, out_dir)
        if not df.empty:
            cross_rows.append(df)

    log_df = _plot_log_suite(out_dir)
    _plot_pal(out_dir)
    _plot_overview(dataset_summaries, cross_rows, log_df, out_dir)

    lines = [
        "# 中文图表包",
        "",
        f"- 输出目录：`{out_dir.resolve()}`",
        "",
        "## 数据集",
    ]
    for item in dataset_summaries:
        lines.append(
            f"- {item['title']}：{item['n_tracks']} 首曲目，{item['n_cultures']} 个文化域，{item['n_users']} 位用户，{item['n_interactions']} 条交互"
        )
    lines.extend(
        [
            "",
            "## 说明",
            "- 这套图表以中文标题、中文坐标名和中文说明为主。",
            "- 专有名词如 Research Dataset V3、CultureMERT、Yambda-5B、LambdaMART、DCAS 保持原文。",
            "- `Yambda-5B` 图表属于补充线，不替代跨文化主线图。",
        ]
    )
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    manifest = {
        "out_dir": str(out_dir.resolve()),
        "dataset_count": len(dataset_summaries),
        "summary_markdown": str((out_dir / "README.md").resolve()),
    }
    (out_dir / "figure_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate a Chinese figure pack for the current project state.")
    ap.add_argument(
        "--out_dir",
        default=str(REPO_ROOT / "reports/figures/project_overview_zh_2026-03-20"),
        help="Output directory for the Chinese figure pack.",
    )
    args = ap.parse_args()
    manifest = generate(Path(str(args.out_dir)))
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
