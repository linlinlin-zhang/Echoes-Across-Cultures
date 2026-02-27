from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from dcas.data.npz_tracks import load_tracks


SPLITS = ("train", "val", "test")


def _round(v: float, ndigits: int = 6) -> float:
    return float(round(float(v), ndigits))


def _normalize_ratios(train: float, val: float, test: float) -> tuple[float, float, float]:
    arr = np.array([float(train), float(val), float(test)], dtype=np.float64)
    if np.any(arr < 0):
        raise ValueError("split ratios must be non-negative")
    s = float(arr.sum())
    if s <= 0:
        raise ValueError("sum of split ratios must be > 0")
    arr = arr / s
    return float(arr[0]), float(arr[1]), float(arr[2])


def _allocate_counts(n: int, ratios: tuple[float, float, float]) -> tuple[int, int, int]:
    if n <= 0:
        return 0, 0, 0
    raw = np.array([ratios[0] * n, ratios[1] * n, ratios[2] * n], dtype=np.float64)
    base = np.floor(raw).astype(np.int64)
    rem = int(n - base.sum())
    frac = raw - base
    order = np.argsort(-frac)
    for i in range(rem):
        base[order[i % len(base)]] += 1
    # always keep at least one train sample when possible
    if base[0] == 0:
        donor = int(np.argmax(base[1:]) + 1)
        if base[donor] > 0:
            base[donor] -= 1
            base[0] += 1
    return int(base[0]), int(base[1]), int(base[2])


def _split_indices_by_culture(
    culture: np.ndarray,
    seed: int,
    ratios: tuple[float, float, float],
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    by_culture: dict[str, list[int]] = defaultdict(list)
    for i, c in enumerate(culture.tolist()):
        by_culture[str(c)].append(int(i))

    out: dict[str, list[int]] = {k: [] for k in SPLITS}
    for c in sorted(by_culture.keys()):
        idx = np.array(by_culture[c], dtype=np.int64)
        rng.shuffle(idx)
        n_tr, n_val, n_te = _allocate_counts(len(idx), ratios)
        tr = idx[:n_tr]
        va = idx[n_tr : n_tr + n_val]
        te = idx[n_tr + n_val : n_tr + n_val + n_te]
        out["train"].extend(tr.tolist())
        out["val"].extend(va.tolist())
        out["test"].extend(te.tolist())

    return {k: np.array(sorted(v), dtype=np.int64) for k, v in out.items()}


def _safe_ratio(num: int, den: int) -> float:
    if den <= 0:
        return 0.0
    return float(num / den)


def _split_distribution_report(
    culture: np.ndarray,
    split_idx: dict[str, np.ndarray],
) -> dict[str, Any]:
    n_total = int(culture.shape[0])
    global_counter = Counter(str(x) for x in culture.tolist())
    all_cultures = sorted(global_counter.keys())

    global_dist = {c: _safe_ratio(global_counter[c], n_total) for c in all_cultures}
    per_split: dict[str, Any] = {}
    max_abs_delta = 0.0

    for sp in SPLITS:
        idx = split_idx[sp]
        c_arr = culture[idx].tolist() if idx.size > 0 else []
        cnt = Counter(str(x) for x in c_arr)
        sp_total = int(len(c_arr))
        rows = []
        for c in all_cultures:
            ratio = _safe_ratio(cnt[c], sp_total)
            delta = abs(ratio - global_dist[c])
            max_abs_delta = max(max_abs_delta, delta)
            rows.append(
                {
                    "culture": c,
                    "count": int(cnt[c]),
                    "ratio": _round(ratio),
                    "delta_from_global": _round(delta),
                }
            )
        per_split[sp] = {
            "n_tracks": int(sp_total),
            "culture_distribution": rows,
        }

    return {
        "global_distribution": [
            {"culture": c, "count": int(global_counter[c]), "ratio": _round(global_dist[c])}
            for c in all_cultures
        ],
        "per_split": per_split,
        "max_abs_delta_from_global": _round(max_abs_delta),
    }


def _load_interactions(path: Path) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _interaction_split_report(
    interactions_path: Path,
    split_track_ids: dict[str, set[str]],
    known_track_ids: set[str],
) -> dict[str, Any]:
    rows = _load_interactions(interactions_path)
    n = len(rows)
    unknown = 0
    per_split_rows: dict[str, int] = {k: 0 for k in SPLITS}
    users_per_split: dict[str, set[str]] = {k: set() for k in SPLITS}
    split_for_tid: dict[str, str] = {}
    for sp in SPLITS:
        for tid in split_track_ids[sp]:
            split_for_tid[tid] = sp

    for r in rows:
        tid = str(r.get("track_id", "")).strip()
        uid = str(r.get("user_id", "")).strip()
        if tid not in known_track_ids:
            unknown += 1
            continue
        sp = split_for_tid.get(tid)
        if sp is None:
            continue
        per_split_rows[sp] += 1
        if uid:
            users_per_split[sp].add(uid)

    return {
        "path": str(interactions_path),
        "n_rows": int(n),
        "unknown_track_count": int(unknown),
        "unknown_track_ratio": _round(_safe_ratio(unknown, n)),
        "per_split_rows": {
            sp: {
                "count": int(per_split_rows[sp]),
                "ratio": _round(_safe_ratio(per_split_rows[sp], n)),
                "n_users": int(len(users_per_split[sp])),
            }
            for sp in SPLITS
        },
    }


def _detect_leakage(split_ids: dict[str, set[str]]) -> dict[str, Any]:
    tr_val = split_ids["train"] & split_ids["val"]
    tr_te = split_ids["train"] & split_ids["test"]
    va_te = split_ids["val"] & split_ids["test"]
    overlap_count = int(len(tr_val) + len(tr_te) + len(va_te))
    return {
        "has_leakage": bool(overlap_count > 0),
        "overlap_count": overlap_count,
        "overlap_samples": {
            "train_val": sorted(list(tr_val))[:20],
            "train_test": sorted(list(tr_te))[:20],
            "val_test": sorted(list(va_te))[:20],
        },
    }


def _to_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Dataset Split Report")
    lines.append("")
    lines.append(f"- status: `{report['status']}`")
    lines.append(f"- n_tracks: `{report['summary']['n_tracks']}`")
    lines.append(f"- seed: `{report['summary']['seed']}`")
    lines.append(
        f"- ratios(train/val/test): `{report['summary']['train_ratio']}/{report['summary']['val_ratio']}/{report['summary']['test_ratio']}`"
    )
    lines.append("")

    lines.append("## Split Sizes")
    lines.append("")
    lines.append("| split | n_tracks | ratio |")
    lines.append("|---|---:|---:|")
    n = max(1, int(report["summary"]["n_tracks"]))
    for sp in SPLITS:
        cnt = int(report["splits"]["counts"][sp])
        lines.append(f"| {sp} | {cnt} | {_round(cnt / n)} |")
    lines.append("")

    lines.append("## Leakage Check")
    lines.append("")
    leak = report["leakage"]
    lines.append(f"- has_leakage: `{leak['has_leakage']}`")
    lines.append(f"- overlap_count: `{leak['overlap_count']}`")
    lines.append("")

    lines.append("## Culture Distribution Drift")
    lines.append("")
    lines.append(f"- max_abs_delta_from_global: `{report['distribution']['max_abs_delta_from_global']}`")
    lines.append("")

    if report.get("interactions"):
        lines.append("## Interaction Coverage")
        lines.append("")
        ir = report["interactions"]
        lines.append(f"- n_rows: `{ir['n_rows']}`")
        lines.append(f"- unknown_track_ratio: `{ir['unknown_track_ratio']}`")
        lines.append("")
        lines.append("| split | interaction_count | ratio | users |")
        lines.append("|---|---:|---:|---:|")
        for sp in SPLITS:
            row = ir["per_split_rows"][sp]
            lines.append(f"| {sp} | {row['count']} | {row['ratio']} | {row['n_users']} |")
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


def make_splits(
    tracks_path: str | Path,
    out_dir: str | Path,
    interactions_path: str | Path | None = None,
    seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    max_distribution_delta_warn: float = 0.1,
) -> dict[str, Any]:
    tracks_p = Path(tracks_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    tracks = load_tracks(str(tracks_p))
    ratios = _normalize_ratios(train_ratio, val_ratio, test_ratio)
    split_idx = _split_indices_by_culture(culture=tracks.culture, seed=seed, ratios=ratios)

    split_ids: dict[str, list[str]] = {}
    split_sets: dict[str, set[str]] = {}
    for sp in SPLITS:
        ids = [str(tracks.track_id[i]) for i in split_idx[sp].tolist()]
        split_ids[sp] = ids
        split_sets[sp] = set(ids)

    leakage = _detect_leakage(split_sets)
    distribution = _split_distribution_report(culture=tracks.culture, split_idx=split_idx)

    issues: list[dict[str, str]] = []
    if leakage["has_leakage"]:
        issues.append(
            {
                "severity": "error",
                "code": "split.leakage",
                "message": f"overlap_count={leakage['overlap_count']}",
            }
        )

    for sp in SPLITS:
        if len(split_ids[sp]) == 0:
            issues.append(
                {
                    "severity": "warn",
                    "code": "split.empty",
                    "message": f"split '{sp}' has zero tracks",
                }
            )

    if float(distribution["max_abs_delta_from_global"]) > float(max_distribution_delta_warn):
        issues.append(
            {
                "severity": "warn",
                "code": "split.distribution_drift",
                "message": (
                    f"max_abs_delta_from_global={distribution['max_abs_delta_from_global']} "
                    f"(>{_round(max_distribution_delta_warn)})"
                ),
            }
        )

    interactions_report: dict[str, Any] | None = None
    if interactions_path is not None:
        inter_p = Path(interactions_path)
        if not inter_p.exists():
            issues.append(
                {
                    "severity": "error",
                    "code": "interactions.not_found",
                    "message": f"interactions file not found: {inter_p}",
                }
            )
        else:
            interactions_report = _interaction_split_report(
                interactions_path=inter_p,
                split_track_ids=split_sets,
                known_track_ids=set(str(x) for x in tracks.track_id.tolist()),
            )
            if interactions_report["unknown_track_ratio"] > 0.0:
                issues.append(
                    {
                        "severity": "error",
                        "code": "interactions.unknown_track",
                        "message": f"unknown_track_ratio={interactions_report['unknown_track_ratio']}",
                    }
                )

    n_errors = sum(1 for x in issues if x["severity"] == "error")
    n_warns = sum(1 for x in issues if x["severity"] == "warn")
    status = "fail" if n_errors > 0 else ("warn" if n_warns > 0 else "pass")

    split_json = out / "split_track_ids.json"
    split_csv = out / "split_assignments.csv"
    report_json = out / "split_report.json"
    report_md = out / "split_report.md"

    with open(split_json, "w", encoding="utf-8") as f:
        json.dump(split_ids, f, ensure_ascii=False, indent=2)

    with open(split_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["track_id", "culture", "split"])
        w.writeheader()
        for sp in SPLITS:
            idxs = split_idx[sp].tolist()
            for i in idxs:
                w.writerow(
                    {
                        "track_id": str(tracks.track_id[i]),
                        "culture": str(tracks.culture[i]),
                        "split": sp,
                    }
                )

    report: dict[str, Any] = {
        "status": status,
        "summary": {
            "tracks_path": str(tracks_p),
            "n_tracks": int(len(tracks)),
            "seed": int(seed),
            "train_ratio": _round(ratios[0]),
            "val_ratio": _round(ratios[1]),
            "test_ratio": _round(ratios[2]),
            "n_errors": int(n_errors),
            "n_warnings": int(n_warns),
        },
        "splits": {
            "counts": {sp: int(len(split_ids[sp])) for sp in SPLITS},
            "outputs": {
                "split_track_ids_json": str(split_json),
                "split_assignments_csv": str(split_csv),
                "split_report_json": str(report_json),
                "split_report_md": str(report_md),
            },
        },
        "leakage": leakage,
        "distribution": distribution,
        "interactions": interactions_report or {},
        "issues": issues,
    }

    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    report_md.write_text(_to_markdown(report), encoding="utf-8")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Create stratified train/val/test splits with leakage checks.")
    ap.add_argument("--tracks", required=True, help="Path to tracks.npz")
    ap.add_argument("--out_dir", required=True, help="Output directory for split artifacts")
    ap.add_argument("--interactions", default=None, help="Optional interactions.csv path")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--test_ratio", type=float, default=0.1)
    ap.add_argument("--max_distribution_delta_warn", type=float, default=0.1)
    ap.add_argument("--strict", action="store_true", help="Exit non-zero when status is fail")
    args = ap.parse_args()

    report = make_splits(
        tracks_path=args.tracks,
        out_dir=args.out_dir,
        interactions_path=args.interactions,
        seed=int(args.seed),
        train_ratio=float(args.train_ratio),
        val_ratio=float(args.val_ratio),
        test_ratio=float(args.test_ratio),
        max_distribution_delta_warn=float(args.max_distribution_delta_warn),
    )
    print(json.dumps({"status": report["status"], "summary": report["summary"]}, ensure_ascii=False))
    if args.strict and report["status"] == "fail":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
