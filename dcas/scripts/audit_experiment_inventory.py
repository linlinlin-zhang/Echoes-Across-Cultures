from __future__ import annotations

import argparse
import csv
import json
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_json(path: Path) -> dict[str, Any] | list[Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve())).replace("\\", "/")
    except Exception:
        return str(path.resolve()).replace("\\", "/")


def _git_status() -> list[str]:
    try:
        out = subprocess.run(
            ["git", "status", "--short", "--branch"],
            cwd=str(REPO_ROOT),
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    except Exception:
        return []
    return [line.rstrip() for line in out.stdout.splitlines() if line.strip()]


def _collect_embedding_manifests() -> list[dict[str, Any]]:
    candidates = [
        REPO_ROOT / "storage/public/research_dataset_v3/tracks_culturemert_v3_main.npz.manifest.json",
        REPO_ROOT / "storage/public/research_dataset_v3/tracks_culturemert_v3_main_mw3.npz.manifest.json",
        REPO_ROOT / "storage/public/research_dataset_v3/tracks_gemini_embedding2_main.npz.manifest.json",
        REPO_ROOT / "storage/public/research_dataset_v3/tracks_gemini_embedding2_v3_main_mw3.npz.manifest.json",
    ]
    rows: list[dict[str, Any]] = []
    for path in candidates:
        obj = _load_json(path)
        if not isinstance(obj, dict):
            continue
        rows.append(
            {
                "path": _rel(path),
                "model_id": obj.get("model_id"),
                "metadata": obj.get("metadata"),
                "out_tracks": obj.get("out_tracks"),
                "n_tracks": obj.get("n_tracks"),
                "dim": obj.get("dim"),
                "max_seconds": obj.get("max_seconds"),
                "window_count": obj.get("window_count", 1),
                "window_strategy": obj.get("window_strategy", "single"),
                "window_aggregate": obj.get("window_aggregate", "mean"),
                "n_errors": obj.get("n_errors", len(obj.get("errors", []))),
            }
        )
    return rows


def _collect_train_configs() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for path in sorted((REPO_ROOT / "configs/train").glob("*.json")):
        obj = _load_json(path)
        if not isinstance(obj, dict):
            continue
        out.append(
            {
                "path": _rel(path),
                "data": obj.get("data"),
                "interactions": obj.get("interactions"),
                "constraints": obj.get("constraints"),
                "out": obj.get("out"),
            }
        )
    return out


def _method_kinds(methods: list[dict[str, Any]], family: str) -> list[str]:
    kinds = [str(m.get("kind", "")) for m in methods if str(m.get("family", "")) == family]
    return sorted({k for k in kinds if k})


def _collect_benchmark_configs() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for path in sorted((REPO_ROOT / "configs/benchmark").glob("*.json")):
        obj = _load_json(path)
        if not isinstance(obj, dict):
            continue
        methods = [m for m in obj.get("methods", []) if isinstance(m, dict)]
        out.append(
            {
                "path": _rel(path),
                "suite_name": obj.get("suite_name"),
                "tracks": obj.get("tracks"),
                "interactions": obj.get("interactions"),
                "n_methods": len(methods),
                "raw_kinds": _method_kinds(methods, "raw"),
                "dcas_kinds": _method_kinds(methods, "dcas"),
            }
        )
    return out


def _collect_benchmark_reports() -> list[dict[str, Any]]:
    root = REPO_ROOT / "reports/benchmarks"
    out: list[dict[str, Any]] = []
    if not root.exists():
        return out
    for path in sorted(p for p in root.iterdir() if p.is_dir()):
        summary_path = path / "benchmark_summary.json"
        table_path = path / "benchmark_table.md"
        summary = _load_json(summary_path)
        method_names: list[str] = []
        if isinstance(summary, dict):
            methods = summary.get("methods")
            if isinstance(methods, dict):
                method_names = sorted(str(k) for k in methods.keys())
        out.append(
            {
                "path": _rel(path),
                "has_summary": summary_path.exists(),
                "has_table": table_path.exists(),
                "methods": method_names,
            }
        )
    return out


def _collect_model_checkpoints() -> list[str]:
    root = REPO_ROOT / "storage/models"
    if not root.exists():
        return []
    return sorted(_rel(p) for p in root.iterdir() if p.is_file())


def _culturemert_mw3_drop_profile() -> dict[str, Any] | None:
    in_path = REPO_ROOT / "storage/public/research_dataset_v3/metadata_v3_main_harmonized.csv"
    out_path = REPO_ROOT / "storage/public/research_dataset_v3/metadata_v3_main_harmonized_mw3.csv"
    if not in_path.exists() or not out_path.exists():
        return None

    with in_path.open("r", encoding="utf-8", newline="") as f:
        rows_in = list(csv.DictReader(f))
    with out_path.open("r", encoding="utf-8", newline="") as f:
        rows_out = list(csv.DictReader(f))

    by_id_in = {str(r.get("track_id", "")).strip(): r for r in rows_in if str(r.get("track_id", "")).strip()}
    ids_out = {str(r.get("track_id", "")).strip() for r in rows_out if str(r.get("track_id", "")).strip()}
    dropped = [row for tid, row in by_id_in.items() if tid not in ids_out]
    if not dropped:
        return {
            "n_dropped": 0,
            "by_culture": {},
            "by_source_dataset": {},
            "examples": [],
        }

    by_culture = Counter(str(r.get("culture", "")).strip() for r in dropped)
    by_source = Counter(str(r.get("source_dataset", "")).strip() for r in dropped)
    examples = [
        {
            "track_id": str(r.get("track_id", "")).strip(),
            "culture": str(r.get("culture", "")).strip(),
            "source_dataset": str(r.get("source_dataset", "")).strip(),
            "audio_path": str(r.get("audio_path", "")).strip(),
        }
        for r in dropped[:16]
    ]
    return {
        "n_dropped": len(dropped),
        "by_culture": dict(sorted(by_culture.items())),
        "by_source_dataset": dict(sorted(by_source.items())),
        "examples": examples,
    }


def _coverage_summary(
    train_configs: list[dict[str, Any]],
    benchmark_configs: list[dict[str, Any]],
    benchmark_reports: list[dict[str, Any]],
) -> dict[str, Any]:
    train_names = {Path(row["path"]).name for row in train_configs}
    bench_names = {Path(row["path"]).name for row in benchmark_configs}
    report_names = {Path(row["path"]).name for row in benchmark_reports}

    expected_gemini = [
        "train_v3_gemini_stage3.run.json",
        "recommender_benchmark_v3_gemini_stage3.run.json",
        "recommender_benchmark_v3_gemini_stage3_bpr.run.json",
        "recommender_benchmark_v3_gemini_stage3_bprhybrid.run.json",
        "recommender_benchmark_v3_gemini_stage3_bprlistwise.run.json",
        "recommender_benchmark_v3_gemini_stage3_dcascal.run.json",
        "recommender_benchmark_v3_gemini_stage3_lambdamart.run.json",
        "recommender_benchmark_v3_gemini_stage3_lightfmlike.run.json",
        "recommender_benchmark_v3_gemini_stage3_stronghybrid.run.json",
        "recommender_benchmark_public_routeA_phase2_cn_gemini.run.json",
    ]

    culture_stage3 = [
        row for row in benchmark_configs if "culturemert_stage3" in Path(row["path"]).name
    ]
    gemini_related = [
        row for row in benchmark_configs if "gemini" in Path(row["path"]).name
    ]

    return {
        "missing_expected_gemini_files": [name for name in expected_gemini if name not in train_names and name not in bench_names],
        "has_any_routeA_gemini_benchmark": any("routeA" in Path(row["path"]).name and "gemini" in Path(row["path"]).name for row in benchmark_configs),
        "culturemert_stage3_benchmark_configs": sorted(Path(row["path"]).name for row in culture_stage3),
        "gemini_benchmark_configs": sorted(Path(row["path"]).name for row in gemini_related),
        "culturemert_stage3_report_dirs": sorted(name for name in report_names if "culturemert_stage3" in name),
        "gemini_report_dirs": sorted(name for name in report_names if "gemini" in name),
    }


def _topline_findings(report: dict[str, Any]) -> list[str]:
    findings: list[str] = []
    coverage = report["coverage"]
    manifests = report["embedding_manifests"]
    profile = report.get("culturemert_mw3_drop_profile")

    if coverage["missing_expected_gemini_files"]:
        findings.append(
            "Gemini 线缺少 stage3 与 public RouteA 对应配置，当前无法与 CultureMERT stage3 做对称对照。"
        )
    for row in manifests:
        if "culturemert_v3_main_mw3" in row["path"] and int(row.get("n_errors") or 0) > 0:
            findings.append(
                f"CultureMERT mw3 embedding 构建存在 {int(row['n_errors'])} 条失败记录，需要在 V4 前修复或单独解释。"
            )
    if isinstance(profile, dict) and int(profile.get("n_dropped", 0)) > 0:
        findings.append(
            f"CultureMERT mw3 对齐后丢失 {int(profile['n_dropped'])} 条 track，存在选择性掉样本风险。"
        )
    return findings


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# 项目实验库存与覆盖自检")
    lines.append("")
    lines.append(f"- 生成时间：`{report['generated_at']}`")
    lines.append("")

    lines.append("## 顶层发现")
    for item in report["topline_findings"]:
        lines.append(f"- {item}")
    if not report["topline_findings"]:
        lines.append("- 未发现明显覆盖缺口。")
    lines.append("")

    lines.append("## Git 状态")
    if report["git_status"]:
        lines.extend(f"- `{line}`" for line in report["git_status"])
    else:
        lines.append("- 无法读取 git 状态。")
    lines.append("")

    lines.append("## Embedding Manifest")
    lines.append("| path | model_id | n_tracks | dim | max_seconds | window_count | n_errors |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for row in report["embedding_manifests"]:
        lines.append(
            f"| {row['path']} | {row.get('model_id')} | {row.get('n_tracks')} | {row.get('dim')} | "
            f"{row.get('max_seconds')} | {row.get('window_count')} | {row.get('n_errors')} |"
        )
    lines.append("")

    lines.append("## Train Config")
    for row in report["train_configs"]:
        lines.append(f"- `{Path(row['path']).name}`")
    lines.append("")

    lines.append("## Benchmark Config")
    lines.append("| config | suite_name | raw_kinds | dcas_kinds |")
    lines.append("|---|---|---|---|")
    for row in report["benchmark_configs"]:
        raw = ", ".join(row["raw_kinds"]) if row["raw_kinds"] else "-"
        dcas = ", ".join(row["dcas_kinds"]) if row["dcas_kinds"] else "-"
        lines.append(f"| {Path(row['path']).name} | {row.get('suite_name')} | {raw} | {dcas} |")
    lines.append("")

    lines.append("## Coverage Summary")
    cov = report["coverage"]
    lines.append(f"- 缺失的 Gemini 对称配置：`{len(cov['missing_expected_gemini_files'])}`")
    for name in cov["missing_expected_gemini_files"]:
        lines.append(f"  - `{name}`")
    lines.append(f"- CultureMERT stage3 report 目录：`{len(cov['culturemert_stage3_report_dirs'])}`")
    lines.append(f"- Gemini report 目录：`{len(cov['gemini_report_dirs'])}`")
    lines.append("")

    profile = report.get("culturemert_mw3_drop_profile")
    if isinstance(profile, dict):
        lines.append("## CultureMERT mw3 掉样本概览")
        lines.append(f"- 掉样本数：`{profile.get('n_dropped', 0)}`")
        if profile.get("by_culture"):
            lines.append("- 按 culture：")
            for key, value in profile["by_culture"].items():
                lines.append(f"  - `{key}`: `{value}`")
        if profile.get("by_source_dataset"):
            lines.append("- 按 source_dataset：")
            for key, value in profile["by_source_dataset"].items():
                lines.append(f"  - `{key}`: `{value}`")
        lines.append("")

    lines.append("## 结果目录")
    for row in report["benchmark_reports"]:
        flags = []
        if row["has_summary"]:
            flags.append("summary")
        if row["has_table"]:
            flags.append("table")
        flag_text = ",".join(flags) if flags else "none"
        lines.append(f"- `{Path(row['path']).name}` [{flag_text}]")
    lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class Args:
    out_dir: str


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit experiment inventory, config coverage, and embedding manifests.")
    ap.add_argument(
        "--out_dir",
        default=str(REPO_ROOT / "reports/audits/project_inventory_2026-03-20"),
        help="Directory for summary artifacts",
    )
    ns = ap.parse_args()
    args = Args(out_dir=str(ns.out_dir))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_configs = _collect_train_configs()
    benchmark_configs = _collect_benchmark_configs()
    benchmark_reports = _collect_benchmark_reports()

    report = {
        "generated_at": subprocess.run(
            ["powershell", "-NoProfile", "-Command", "Get-Date -Format o"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
        ).stdout.strip(),
        "git_status": _git_status(),
        "embedding_manifests": _collect_embedding_manifests(),
        "train_configs": train_configs,
        "benchmark_configs": benchmark_configs,
        "benchmark_reports": benchmark_reports,
        "model_checkpoints": _collect_model_checkpoints(),
        "coverage": _coverage_summary(
            train_configs=train_configs,
            benchmark_configs=benchmark_configs,
            benchmark_reports=benchmark_reports,
        ),
        "culturemert_mw3_drop_profile": _culturemert_mw3_drop_profile(),
    }
    report["topline_findings"] = _topline_findings(report)

    summary_json = out_dir / "summary.json"
    summary_md = out_dir / "summary.md"
    summary_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_markdown(summary_md, report)

    print(str(summary_json))
    print(str(summary_md))


if __name__ == "__main__":
    main()
