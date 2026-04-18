from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _parse_similar(value: str) -> bool | None:
    v = str(value).strip().lower()
    if v in {"1", "true", "yes", "y", "similar", "sim"}:
        return True
    if v in {"0", "false", "no", "n", "dissimilar", "diff"}:
        return False
    return None


def _parse_judgment(value: str) -> bool | None:
    v = str(value).strip().lower()
    if v in {"a", "similar", "same", "yes", "y"}:
        return True
    if v in {"b", "dissimilar", "different", "diff", "no", "n"}:
        return False
    if v in {"", "neither", "uncertain", "unsure"}:
        return None
    return None


def build_constraints_from_annotations(
    annotations_csv: str,
    out_path: str,
    conflict_policy: str = "last",
    report_path: str | None = None,
) -> dict[str, object]:
    policy = str(conflict_policy).strip().lower()
    if policy not in {"last", "first", "drop"}:
        raise ValueError("conflict_policy must be one of: last, first, drop")

    rows: list[dict[str, object]] = []
    skipped = 0
    with open(annotations_csv, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for line_no, row in enumerate(reader, start=2):
            a = str(row.get("track_id_a", "")).strip()
            b = str(row.get("track_id_b", "")).strip()
            judgment = _parse_judgment(str(row.get("judgment", "")))
            similar = judgment if judgment is not None else _parse_similar(str(row.get("similar", "")))
            rationale = str(row.get("rationale", "")).strip()
            if not a or not b or a == b or similar is None:
                skipped += 1
                continue
            key_a, key_b = (a, b) if a < b else (b, a)
            rows.append(
                {
                    "track_id_a": key_a,
                    "track_id_b": key_b,
                    "similar": bool(similar),
                    "rationale": rationale,
                    "task_id": str(row.get("task_id", "")).strip(),
                    "line_no": int(line_no),
                }
            )

    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault((str(row["track_id_a"]), str(row["track_id_b"])), []).append(row)

    merged: list[dict[str, object]] = []
    duplicate_pair_count = 0
    consistent_duplicate_pairs = 0
    conflicting_duplicate_pairs = 0
    dropped_conflicting_pairs = 0
    conflict_examples: list[dict[str, object]] = []
    for key, items in grouped.items():
        if len(items) > 1:
            duplicate_pair_count += 1
        labels = {bool(item["similar"]) for item in items}
        if len(items) > 1 and len(labels) == 1:
            consistent_duplicate_pairs += 1
        if len(labels) > 1:
            conflicting_duplicate_pairs += 1
            if len(conflict_examples) < 20:
                conflict_examples.append(
                    {
                        "track_id_a": str(key[0]),
                        "track_id_b": str(key[1]),
                        "task_ids": [str(item["task_id"]) for item in items],
                        "labels": [bool(item["similar"]) for item in items],
                    }
                )
            if policy == "drop":
                dropped_conflicting_pairs += 1
                continue
            chosen = items[0] if policy == "first" else items[-1]
        else:
            chosen = items[-1]
        merged.append(
            {
                "track_id_a": str(chosen["track_id_a"]),
                "track_id_b": str(chosen["track_id_b"]),
                "similar": bool(chosen["similar"]),
                "rationale": str(chosen["rationale"]),
            }
        )
    merged.sort(key=lambda r: (str(r["track_id_a"]), str(r["track_id_b"])))

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for row in merged:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    n_positive = sum(1 for row in merged if bool(row["similar"]))
    report = {
        "annotations_csv": str(annotations_csv),
        "out_path": str(out),
        "n_constraints": int(len(merged)),
        "n_positive": int(n_positive),
        "n_negative": int(len(merged) - n_positive),
        "n_skipped": int(skipped),
        "conflict_policy": str(policy),
        "duplicate_pair_count": int(duplicate_pair_count),
        "consistent_duplicate_pairs": int(consistent_duplicate_pairs),
        "conflicting_duplicate_pairs": int(conflicting_duplicate_pairs),
        "dropped_conflicting_pairs": int(dropped_conflicting_pairs),
        "conflict_examples_preview": conflict_examples,
    }
    if report_path is not None:
        rep = Path(report_path)
        rep.parent.mkdir(parents=True, exist_ok=True)
        with open(rep, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Build pairwise constraints jsonl from annotated PAL CSV.")
    ap.add_argument("--annotations", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--conflict_policy", default="last", choices=["last", "first", "drop"])
    ap.add_argument("--report_json", default=None)
    args = ap.parse_args()

    rep = build_constraints_from_annotations(
        annotations_csv=str(args.annotations),
        out_path=str(args.out),
        conflict_policy=str(args.conflict_policy),
        report_path=str(args.report_json) if args.report_json else None,
    )
    print(json.dumps(rep, ensure_ascii=False))


if __name__ == "__main__":
    main()
