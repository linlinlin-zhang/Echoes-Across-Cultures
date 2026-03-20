from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _git_status() -> str:
    try:
        out = subprocess.run(
            ["git", "status", "--short", "--branch"],
            cwd=str(REPO_ROOT),
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        return str(out.stdout).strip()
    except Exception as e:
        return f"git status unavailable: {e}"


def _paper_placeholder_flags(path: Path) -> dict[str, bool]:
    if not path.exists():
        return {"exists": False}
    text = path.read_text(encoding="utf-8", errors="ignore").lower()
    return {
        "exists": True,
        "contains_placeholder": "placeholder" in text,
        "contains_draft_evaluation": "draft evaluation" in text,
        "contains_four_domain_1600": ("1600 tracks" in text) or ("four cultural domains" in text),
        "contains_synthetic_placeholder_outcomes": "synthetic placeholder outcomes" in text,
    }


def _bench_status() -> list[dict[str, Any]]:
    suites = [
        ("v3_main_culturemert", "reports/benchmarks/v3_main_culturemert/benchmark_summary.json"),
        ("v3_main_culturemert_stage3", "reports/benchmarks/v3_main_culturemert_stage3/benchmark_summary.json"),
        (
            "v3_main_culturemert_stage3_lambdamart",
            "reports/benchmarks/v3_main_culturemert_stage3_lambdamart/benchmark_summary.json",
        ),
        ("v3_main_gemini_embedding2", "reports/benchmarks/v3_main_gemini_embedding2/benchmark_summary.json"),
        (
            "public_routeA_phase2_cn_lambdamart",
            "reports/benchmarks/public_routeA_phase2_cn_lambdamart/benchmark_summary.json",
        ),
        (
            "yambda_5b_subset_global_log_benchmark",
            "reports/benchmarks/yambda_5b_subset_global_log_benchmark/benchmark_summary.json",
        ),
        ("v3_main_gemini_stage3_expected", "reports/benchmarks/v3_main_gemini_stage3/benchmark_summary.json"),
        (
            "v3_main_gemini_stage3_lambdamart_expected",
            "reports/benchmarks/v3_main_gemini_stage3_lambdamart/benchmark_summary.json",
        ),
        (
            "public_routeA_phase2_cn_gemini_expected",
            "reports/benchmarks/public_routeA_phase2_cn_gemini/benchmark_summary.json",
        ),
        ("mssd_expected", "reports/benchmarks/mssd/benchmark_summary.json"),
    ]
    rows: list[dict[str, Any]] = []
    for name, rel in suites:
        path = REPO_ROOT / rel
        payload = _read_json(path)
        rows.append(
            {
                "suite": name,
                "path": str(path),
                "exists": path.exists(),
                "method_count": int(len(payload.get("methods", {}))) if payload else None,
            }
        )
    return rows


def audit_project_state() -> dict[str, Any]:
    findings: list[dict[str, str]] = []

    v3_summary = _read_json(REPO_ROOT / "storage/public/research_dataset_v3/summary_v3_main.json") or {}
    culturemert_manifest = (
        _read_json(REPO_ROOT / "storage/public/research_dataset_v3/tracks_culturemert_v3_main_mw3.npz.manifest.json")
        or {}
    )
    mw3_align_report = (
        _read_json(REPO_ROOT / "storage/public/research_dataset_v3/metadata_v3_main_harmonized_mw3.csv.align_report.json")
        or {}
    )
    paper_flags = _paper_placeholder_flags(REPO_ROOT / "paper/ismir2026_draft.tex")
    bench_rows = _bench_status()
    git_status = _git_status()

    domains = list(v3_summary.get("domains", []))
    single_source_domains = [str(d.get("culture")) for d in domains if len(list(d.get("sources", []))) <= 1]
    zero_artist_domains = [str(d.get("culture")) for d in domains if int(d.get("n_artists", 0) or 0) == 0]

    if paper_flags.get("contains_placeholder"):
        findings.append(
            {
                "severity": "warn",
                "code": "paper.placeholder_content",
                "message": "paper/ismir2026_draft.tex still contains placeholder wording and should be synchronized with real experiments.",
            }
        )
    if paper_flags.get("contains_four_domain_1600"):
        findings.append(
            {
                "severity": "warn",
                "code": "paper.outdated_dataset_description",
                "message": "paper draft still references the older four-domain/1600-track setup instead of the current V3/routeA evidence structure.",
            }
        )
    if int(culturemert_manifest.get("n_errors", 0) or 0) > 0:
        findings.append(
            {
                "severity": "warn",
                "code": "culturemert.embedding_failures",
                "message": f"CultureMERT mw3 embedding build dropped {int(culturemert_manifest.get('n_errors', 0))} rows and needs audit or recovery.",
            }
        )
    if int(mw3_align_report.get("metadata_rows_dropped", 0) or 0) > 0:
        findings.append(
            {
                "severity": "warn",
                "code": "dataset.mw3_alignment_drop",
                "message": f"mw3 alignment dropped {int(mw3_align_report.get('metadata_rows_dropped', 0))} metadata rows and {int(mw3_align_report.get('interactions_rows_dropped', 0) or 0)} interactions.",
            }
        )
    if single_source_domains:
        findings.append(
            {
                "severity": "warn",
                "code": "dataset.source_confound_risk",
                "message": f"V3 has cultures dominated by a single source dataset: {', '.join(single_source_domains)}.",
            }
        )
    if zero_artist_domains:
        findings.append(
            {
                "severity": "info",
                "code": "dataset.metadata_sparse_artist",
                "message": f"Some V3 cultures still have zero artist metadata coverage: {', '.join(zero_artist_domains)}.",
            }
        )

    missing_expected = [row["suite"] for row in bench_rows if str(row["suite"]).endswith("_expected") and not row["exists"]]
    if missing_expected:
        findings.append(
            {
                "severity": "warn",
                "code": "benchmark.matrix_incomplete",
                "message": f"Expected benchmark lines are still missing: {', '.join(missing_expected)}.",
            }
        )

    if not any(row["suite"] == "yambda_5b_subset_global_log_benchmark" and row["exists"] for row in bench_rows):
        findings.append(
            {
                "severity": "warn",
                "code": "benchmark.external_log_missing",
                "message": "Yambda subset log benchmark output is missing.",
            }
        )

    if not any(row["suite"] == "mssd_expected" and row["exists"] for row in bench_rows):
        findings.append(
            {
                "severity": "info",
                "code": "benchmark.mssd_missing",
                "message": "MSSD benchmark artifacts are absent; current repo evidence still lacks that external log line.",
            }
        )

    return {
        "repo_root": str(REPO_ROOT),
        "git_status": git_status,
        "paper": paper_flags,
        "datasets": {
            "v3_tracks": int(sum(int(d.get("n_rows", 0) or 0) for d in domains)),
            "v3_cultures": [str(d.get("culture")) for d in domains],
            "single_source_domains": single_source_domains,
            "zero_artist_domains": zero_artist_domains,
            "culturemert_mw3_manifest": culturemert_manifest,
            "mw3_align_report": mw3_align_report,
        },
        "benchmarks": bench_rows,
        "findings": findings,
    }


def _to_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.extend(["# Project Self Audit", "", "## Repo", ""])
    lines.append("```text")
    lines.append(str(report.get("git_status", "")).strip())
    lines.append("```")
    lines.extend(["", "## Key Findings", ""])
    findings = list(report.get("findings", []))
    if not findings:
        lines.append("- none")
    else:
        lines.append("| severity | code | message |")
        lines.append("|---|---|---|")
        for row in findings:
            lines.append(f"| {row['severity']} | {row['code']} | {row['message']} |")

    lines.extend(["", "## Dataset Signals", ""])
    datasets = dict(report.get("datasets", {}))
    lines.append(f"- V3 cultures: `{len(list(datasets.get('v3_cultures', [])))}`")
    lines.append(f"- Single-source cultures: `{', '.join(datasets.get('single_source_domains', [])) or 'none'}`")
    lines.append(f"- Zero-artist cultures: `{', '.join(datasets.get('zero_artist_domains', [])) or 'none'}`")
    manifest = dict(datasets.get("culturemert_mw3_manifest", {}))
    align = dict(datasets.get("mw3_align_report", {}))
    lines.append(f"- CultureMERT mw3 embedding errors: `{int(manifest.get('n_errors', 0) or 0)}`")
    lines.append(f"- mw3 metadata rows dropped: `{int(align.get('metadata_rows_dropped', 0) or 0)}`")
    lines.append(f"- mw3 interaction rows dropped: `{int(align.get('interactions_rows_dropped', 0) or 0)}`")

    lines.extend(["", "## Benchmark Matrix", "", "| suite | exists | method_count |", "|---|---|---:|"])
    for row in report.get("benchmarks", []):
        lines.append(f"| {row['suite']} | {str(bool(row['exists'])).lower()} | {row['method_count'] if row['method_count'] is not None else ''} |")

    lines.extend(["", "## Paper Audit", "", "| flag | value |", "|---|---|"])
    for key, value in dict(report.get("paper", {})).items():
        lines.append(f"| {key} | {value} |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit project state for scientific rigor and reproducibility gaps.")
    ap.add_argument(
        "--out_dir",
        default=str(REPO_ROOT / "reports" / "audits" / "project_self_audit_2026-03-20"),
        help="Directory for markdown/json outputs",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report = audit_project_state()
    (out_dir / "audit_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "audit_report.md").write_text(_to_markdown(report), encoding="utf-8")
    print(json.dumps({"out_dir": str(out_dir), "findings": len(report.get("findings", []))}, ensure_ascii=False))


if __name__ == "__main__":
    main()
