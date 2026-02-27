from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from dcas.data.npz_tracks import load_tracks


@dataclass(frozen=True)
class ValidationThresholds:
    min_tracks_per_culture: int = 30
    max_culture_imbalance_ratio: float = 20.0
    max_unknown_track_ratio: float = 0.01
    max_duplicate_user_track_ratio: float = 0.05
    max_zero_norm_ratio: float = 0.05
    min_interactions_per_user: int = 5


def _safe_float(v: Any) -> float | None:
    try:
        f = float(v)
    except Exception:
        return None
    if not math.isfinite(f):
        return None
    return f


def _round(v: float, ndigits: int = 6) -> float:
    return float(round(float(v), ndigits))


def _norm_stats(emb: np.ndarray) -> dict[str, float]:
    norms = np.linalg.norm(emb, axis=1)
    q = np.quantile(norms, [0.0, 0.25, 0.5, 0.75, 1.0])
    return {
        "min": _round(float(q[0])),
        "p25": _round(float(q[1])),
        "p50": _round(float(q[2])),
        "p75": _round(float(q[3])),
        "max": _round(float(q[4])),
        "mean": _round(float(norms.mean())),
        "std": _round(float(norms.std())),
    }


def _to_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Dataset Profile Report")
    lines.append("")
    lines.append(f"- status: `{report['status']}`")
    lines.append(f"- tracks: `{report['summary']['n_tracks']}`")
    lines.append(f"- embedding_dim: `{report['summary']['embedding_dim']}`")
    lines.append(f"- cultures: `{report['summary']['n_cultures']}`")
    if report["summary"]["has_interactions"]:
        lines.append(f"- interactions: `{report['summary']['n_interactions']}`")
        lines.append(f"- users: `{report['summary']['n_users']}`")
    lines.append("")

    lines.append("## Tracks")
    lines.append("")
    tr = report["tracks"]
    lines.append(f"- duplicate_track_ids: `{tr['duplicate_track_ids']}`")
    lines.append(f"- finite_embedding_ratio: `{tr['finite_embedding_ratio']}`")
    lines.append(f"- zero_norm_ratio: `{tr['zero_norm_ratio']}`")
    lines.append(f"- culture_imbalance_ratio: `{tr['culture_imbalance_ratio']}`")
    lines.append("")
    lines.append("### Culture Distribution")
    lines.append("")
    lines.append("| culture | count | ratio |")
    lines.append("|---|---:|---:|")
    for c in tr["culture_distribution"]:
        lines.append(f"| {c['culture']} | {c['count']} | {c['ratio']} |")
    lines.append("")

    if report["summary"]["has_interactions"]:
        ir = report["interactions"]
        lines.append("## Interactions")
        lines.append("")
        lines.append(f"- unknown_track_ratio: `{ir['unknown_track_ratio']}`")
        lines.append(f"- duplicate_user_track_ratio: `{ir['duplicate_user_track_ratio']}`")
        lines.append(f"- non_positive_weight_ratio: `{ir['non_positive_weight_ratio']}`")
        lines.append(f"- track_coverage_ratio: `{ir['track_coverage_ratio']}`")
        lines.append("")

    lines.append("## Issues")
    lines.append("")
    if not report["issues"]:
        lines.append("- none")
    else:
        lines.append("| severity | code | message |")
        lines.append("|---|---|---|")
        for it in report["issues"]:
            lines.append(f"| {it['severity']} | {it['code']} | {it['message']} |")
    lines.append("")
    return "\n".join(lines)


def _validate_tracks(path: Path, th: ValidationThresholds) -> tuple[dict[str, Any], list[dict[str, str]], set[str]]:
    issues: list[dict[str, str]] = []
    tracks = load_tracks(str(path))
    n = len(tracks)
    if n == 0:
        issues.append({"severity": "error", "code": "tracks.empty", "message": "tracks.npz has zero tracks"})
    emb = tracks.embedding

    finite_mask = np.isfinite(emb)
    finite_ratio = float(finite_mask.mean()) if emb.size > 0 else 0.0
    if finite_ratio < 1.0:
        issues.append(
            {
                "severity": "error",
                "code": "tracks.embedding_non_finite",
                "message": f"embedding contains non-finite values (finite_ratio={_round(finite_ratio)})",
            }
        )

    norm = np.linalg.norm(emb, axis=1) if n > 0 else np.array([], dtype=np.float32)
    zero_norm_ratio = float((norm <= 1e-12).mean()) if n > 0 else 0.0
    if zero_norm_ratio > th.max_zero_norm_ratio:
        issues.append(
            {
                "severity": "warn",
                "code": "tracks.zero_norm_high",
                "message": f"zero-norm embedding ratio is high ({_round(zero_norm_ratio)})",
            }
        )

    ids = [str(x) for x in tracks.track_id.tolist()]
    id_counter = Counter(ids)
    duplicate_track_ids = int(sum(1 for _, cnt in id_counter.items() if cnt > 1))
    if duplicate_track_ids > 0:
        issues.append(
            {
                "severity": "error",
                "code": "tracks.duplicate_track_id",
                "message": f"duplicate track_id found ({duplicate_track_ids})",
            }
        )

    culture_counter = Counter(str(x) for x in tracks.culture.tolist())
    culture_rows = []
    for c, cnt in sorted(culture_counter.items(), key=lambda t: (-t[1], t[0])):
        ratio = float(cnt / max(1, n))
        culture_rows.append({"culture": c, "count": int(cnt), "ratio": _round(ratio)})
        if cnt < th.min_tracks_per_culture:
            issues.append(
                {
                    "severity": "warn",
                    "code": "tracks.culture_low_count",
                    "message": f"culture '{c}' has only {cnt} tracks (< {th.min_tracks_per_culture})",
                }
            )

    if culture_counter:
        min_c = min(culture_counter.values())
        max_c = max(culture_counter.values())
        imb = float(max_c / max(1, min_c))
    else:
        imb = 0.0
    if imb > th.max_culture_imbalance_ratio:
        issues.append(
            {
                "severity": "warn",
                "code": "tracks.culture_imbalance",
                "message": f"culture imbalance ratio is high ({_round(imb)})",
            }
        )

    affect_stats: dict[str, Any]
    if tracks.affect_label is None:
        affect_stats = {"present": False}
        issues.append(
            {
                "severity": "info",
                "code": "tracks.affect_missing",
                "message": "affect_label is absent (allowed, but limits affect-related evaluation)",
            }
        )
    else:
        ac = Counter(int(x) for x in tracks.affect_label.tolist())
        affect_stats = {
            "present": True,
            "n_classes": int(len(ac)),
            "distribution": [{"label": int(k), "count": int(v)} for k, v in sorted(ac.items())],
        }

    out = {
        "path": str(path),
        "n_tracks": int(n),
        "embedding_dim": int(tracks.dim),
        "duplicate_track_ids": int(duplicate_track_ids),
        "finite_embedding_ratio": _round(finite_ratio),
        "zero_norm_ratio": _round(zero_norm_ratio),
        "embedding_norm_stats": _norm_stats(emb) if n > 0 else {},
        "culture_distribution": culture_rows,
        "culture_imbalance_ratio": _round(imb),
        "affect": affect_stats,
    }
    return out, issues, set(ids)


def _validate_interactions(
    path: Path,
    known_track_ids: set[str],
    n_tracks_total: int,
    th: ValidationThresholds,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    issues: list[dict[str, str]] = []
    if not path.exists():
        issues.append(
            {
                "severity": "error",
                "code": "interactions.not_found",
                "message": f"interactions file not found: {path}",
            }
        )
        return {"path": str(path), "n_interactions": 0, "n_users": 0}, issues

    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fields = set(reader.fieldnames or [])
        required = {"user_id", "track_id"}
        missing = sorted(list(required - fields))
        if missing:
            issues.append(
                {
                    "severity": "error",
                    "code": "interactions.missing_columns",
                    "message": f"missing required columns: {missing}",
                }
            )
            return {"path": str(path), "n_interactions": 0, "n_users": 0}, issues

        rows = list(reader)

    n_rows = len(rows)
    if n_rows == 0:
        issues.append(
            {
                "severity": "error",
                "code": "interactions.empty",
                "message": "interactions.csv has zero rows",
            }
        )

    unknown_track = 0
    missing_user = 0
    missing_track = 0
    invalid_weight = 0
    non_positive_weight = 0

    user_counter: Counter[str] = Counter()
    track_counter: Counter[str] = Counter()
    user_track_counter: Counter[tuple[str, str]] = Counter()

    for row in rows:
        user = str(row.get("user_id", "")).strip()
        tid = str(row.get("track_id", "")).strip()
        if not user:
            missing_user += 1
            continue
        if not tid:
            missing_track += 1
            continue
        user_counter[user] += 1
        track_counter[tid] += 1
        user_track_counter[(user, tid)] += 1
        if tid not in known_track_ids:
            unknown_track += 1
        w = _safe_float(row.get("weight", 1.0))
        if w is None:
            invalid_weight += 1
        elif w <= 0:
            non_positive_weight += 1

    if missing_user > 0 or missing_track > 0:
        issues.append(
            {
                "severity": "error",
                "code": "interactions.missing_keys",
                "message": f"rows with missing user_id/track_id: {missing_user + missing_track}",
            }
        )

    if invalid_weight > 0:
        issues.append(
            {
                "severity": "error",
                "code": "interactions.invalid_weight",
                "message": f"rows with invalid non-finite weight: {invalid_weight}",
            }
        )

    dup_user_track = int(sum(1 for _, cnt in user_track_counter.items() if cnt > 1))
    dup_ratio = float(dup_user_track / max(1, len(user_track_counter)))
    if dup_ratio > th.max_duplicate_user_track_ratio:
        issues.append(
            {
                "severity": "warn",
                "code": "interactions.duplicate_user_track_high",
                "message": f"duplicate (user,track) ratio is high ({_round(dup_ratio)})",
            }
        )

    unknown_ratio = float(unknown_track / max(1, n_rows))
    if unknown_ratio > th.max_unknown_track_ratio:
        issues.append(
            {
                "severity": "error",
                "code": "interactions.unknown_track_high",
                "message": f"unknown track ratio is high ({_round(unknown_ratio)})",
            }
        )

    if user_counter:
        min_user = min(user_counter.values())
        if min_user < th.min_interactions_per_user:
            issues.append(
                {
                    "severity": "warn",
                    "code": "interactions.low_user_activity",
                    "message": f"some users have fewer than {th.min_interactions_per_user} interactions",
                }
            )

    covered_tracks = int(sum(1 for t in track_counter if t in known_track_ids))
    coverage_ratio = float(covered_tracks / max(1, n_tracks_total))

    out = {
        "path": str(path),
        "n_interactions": int(n_rows),
        "n_users": int(len(user_counter)),
        "unknown_track_count": int(unknown_track),
        "unknown_track_ratio": _round(unknown_ratio),
        "duplicate_user_track_count": int(dup_user_track),
        "duplicate_user_track_ratio": _round(dup_ratio),
        "invalid_weight_count": int(invalid_weight),
        "non_positive_weight_count": int(non_positive_weight),
        "non_positive_weight_ratio": _round(float(non_positive_weight / max(1, n_rows))),
        "track_coverage_ratio": _round(coverage_ratio),
        "user_activity_stats": {
            "min": int(min(user_counter.values())) if user_counter else 0,
            "max": int(max(user_counter.values())) if user_counter else 0,
            "mean": _round(float(np.mean(list(user_counter.values())))) if user_counter else 0.0,
            "median": _round(float(np.median(list(user_counter.values())))) if user_counter else 0.0,
        },
    }
    return out, issues


def validate_dataset(
    tracks_path: str | Path,
    interactions_path: str | Path | None = None,
    thresholds: ValidationThresholds | None = None,
) -> dict[str, Any]:
    th = thresholds or ValidationThresholds()
    tracks_p = Path(tracks_path)

    issues: list[dict[str, str]] = []
    if not tracks_p.exists():
        raise FileNotFoundError(f"tracks file not found: {tracks_p}")

    tracks_info, track_issues, known_track_ids = _validate_tracks(tracks_p, th)
    issues.extend(track_issues)

    has_interactions = interactions_path is not None
    interactions_info: dict[str, Any] | None = None
    if has_interactions:
        inter_p = Path(interactions_path)
        interactions_info, inter_issues = _validate_interactions(
            path=inter_p,
            known_track_ids=known_track_ids,
            n_tracks_total=int(tracks_info["n_tracks"]),
            th=th,
        )
        issues.extend(inter_issues)

    n_errors = sum(1 for i in issues if i["severity"] == "error")
    n_warns = sum(1 for i in issues if i["severity"] == "warn")
    status = "fail" if n_errors > 0 else ("warn" if n_warns > 0 else "pass")

    summary = {
        "n_tracks": int(tracks_info["n_tracks"]),
        "embedding_dim": int(tracks_info["embedding_dim"]),
        "n_cultures": int(len(tracks_info["culture_distribution"])),
        "has_interactions": bool(has_interactions),
        "n_interactions": int(interactions_info["n_interactions"]) if interactions_info else 0,
        "n_users": int(interactions_info["n_users"]) if interactions_info else 0,
        "n_issues": int(len(issues)),
        "n_errors": int(n_errors),
        "n_warnings": int(n_warns),
    }

    report: dict[str, Any] = {
        "status": status,
        "summary": summary,
        "tracks": tracks_info,
        "interactions": interactions_info or {},
        "thresholds": {
            "min_tracks_per_culture": th.min_tracks_per_culture,
            "max_culture_imbalance_ratio": th.max_culture_imbalance_ratio,
            "max_unknown_track_ratio": th.max_unknown_track_ratio,
            "max_duplicate_user_track_ratio": th.max_duplicate_user_track_ratio,
            "max_zero_norm_ratio": th.max_zero_norm_ratio,
            "min_interactions_per_user": th.min_interactions_per_user,
        },
        "issues": issues,
    }
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate dataset quality and generate profile reports.")
    ap.add_argument("--tracks", required=True, help="Path to tracks.npz")
    ap.add_argument("--interactions", default=None, help="Path to interactions.csv")
    ap.add_argument("--out_json", default=None, help="Output report json path")
    ap.add_argument("--out_md", default=None, help="Output report markdown path")
    ap.add_argument("--strict", action="store_true", help="Exit non-zero when report status is fail")
    ap.add_argument("--min_tracks_per_culture", type=int, default=30)
    ap.add_argument("--max_culture_imbalance_ratio", type=float, default=20.0)
    ap.add_argument("--max_unknown_track_ratio", type=float, default=0.01)
    ap.add_argument("--max_duplicate_user_track_ratio", type=float, default=0.05)
    ap.add_argument("--max_zero_norm_ratio", type=float, default=0.05)
    ap.add_argument("--min_interactions_per_user", type=int, default=5)
    args = ap.parse_args()

    thresholds = ValidationThresholds(
        min_tracks_per_culture=int(args.min_tracks_per_culture),
        max_culture_imbalance_ratio=float(args.max_culture_imbalance_ratio),
        max_unknown_track_ratio=float(args.max_unknown_track_ratio),
        max_duplicate_user_track_ratio=float(args.max_duplicate_user_track_ratio),
        max_zero_norm_ratio=float(args.max_zero_norm_ratio),
        min_interactions_per_user=int(args.min_interactions_per_user),
    )

    report = validate_dataset(
        tracks_path=args.tracks,
        interactions_path=args.interactions,
        thresholds=thresholds,
    )

    if args.out_json:
        out_json = Path(args.out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
    if args.out_md:
        out_md = Path(args.out_md)
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(_to_markdown(report), encoding="utf-8")

    print(json.dumps({"status": report["status"], "summary": report["summary"]}, ensure_ascii=False))
    if args.strict and report["status"] == "fail":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
