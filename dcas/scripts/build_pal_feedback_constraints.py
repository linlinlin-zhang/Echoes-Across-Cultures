from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _load_track_labels(metadata_csv: str, track_id_col: str, label_col: str) -> dict[str, str]:
    out: dict[str, str] = {}
    with open(metadata_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tid = str(row.get(track_id_col, "")).strip()
            label = str(row.get(label_col, "")).strip()
            if tid:
                out[tid] = label
    return out


def build_constraints(
    tasks_path: str,
    metadata_csv: str,
    out_path: str,
    track_id_col: str = "track_id",
    label_col: str = "label",
) -> dict[str, Any]:
    labels = _load_track_labels(metadata_csv=metadata_csv, track_id_col=track_id_col, label_col=label_col)
    seen_pairs: set[tuple[str, str]] = set()
    rows: list[dict[str, Any]] = []
    skipped = 0

    with open(tasks_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            a = str(obj.get("track_id", "")).strip()
            b = str(obj.get("compare_to", "")).strip()
            if not a or not b or a == b:
                skipped += 1
                continue
            if a not in labels or b not in labels:
                skipped += 1
                continue
            key = (a, b) if a < b else (b, a)
            if key in seen_pairs:
                continue
            seen_pairs.add(key)

            la = labels[a]
            lb = labels[b]
            similar = bool(la == lb and la != "")
            if similar:
                rationale = f"simulated expert: same label ({label_col})={la}"
            else:
                rationale = f"simulated expert: different label ({label_col}) {la} vs {lb}"

            rows.append(
                {
                    "track_id_a": key[0],
                    "track_id_b": key[1],
                    "similar": similar,
                    "rationale": rationale,
                }
            )

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_pos = sum(1 for r in rows if bool(r["similar"]))
    n_neg = len(rows) - n_pos
    return {
        "tasks_path": str(tasks_path),
        "metadata_csv": str(metadata_csv),
        "out_path": str(out),
        "n_constraints": int(len(rows)),
        "n_positive": int(n_pos),
        "n_negative": int(n_neg),
        "n_skipped": int(skipped),
        "track_id_col": str(track_id_col),
        "label_col": str(label_col),
        "mode": "simulated_expert_same_label",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Build pairwise constraints from PAL tasks and metadata labels.")
    ap.add_argument("--tasks", required=True, help="pal tasks jsonl, each row has track_id and compare_to")
    ap.add_argument("--metadata", required=True, help="metadata csv containing track_id and label columns")
    ap.add_argument("--out", required=True, help="output constraints jsonl")
    ap.add_argument("--track_id_col", default="track_id")
    ap.add_argument("--label_col", default="label")
    ap.add_argument("--report_json", default=None)
    args = ap.parse_args()

    rep = build_constraints(
        tasks_path=str(args.tasks),
        metadata_csv=str(args.metadata),
        out_path=str(args.out),
        track_id_col=str(args.track_id_col),
        label_col=str(args.label_col),
    )
    if args.report_json:
        out = Path(str(args.report_json))
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(rep, f, ensure_ascii=False, indent=2)
    print(json.dumps(rep, ensure_ascii=False))


if __name__ == "__main__":
    main()

