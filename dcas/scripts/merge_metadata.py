from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fields = list(reader.fieldnames or [])
    return rows, fields


def merge_metadata(
    inputs: list[str | Path],
    out_csv: str | Path,
) -> dict[str, Any]:
    in_paths = [Path(p) for p in inputs]
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    merged_rows: list[dict[str, str]] = []
    all_fields: list[str] = []
    seen_fields: set[str] = set()
    sources: list[dict[str, Any]] = []

    for p in in_paths:
        rows, fields = _read_csv(p)
        required = {"track_id", "culture", "audio_path"}
        missing = sorted(list(required - set(fields)))
        if missing:
            raise RuntimeError(f"metadata missing required columns at {p}: {missing}")
        for c in fields:
            if c not in seen_fields:
                seen_fields.add(c)
                all_fields.append(c)
        n_before = len(merged_rows)
        for r in rows:
            rr = dict(r)
            rel = str(rr.get("audio_path", "")).strip()
            if rel == "":
                raise RuntimeError(f"empty audio_path in {p}")
            ap = Path(rel)
            if not ap.is_absolute():
                ap = (p.parent / ap).resolve()
            rr["audio_path"] = str(ap)
            merged_rows.append(rr)
        sources.append({"path": str(p.resolve()), "rows": int(len(merged_rows) - n_before)})

    preferred = ["track_id", "culture", "audio_path", "source_dataset", "source_split", "source_index", "label", "affect_label"]
    extra = [c for c in all_fields if c not in preferred]
    cols = [c for c in preferred if c in seen_fields] + extra

    track_counter = Counter(str(r.get("track_id", "")).strip() for r in merged_rows)
    dup = sorted([k for k, v in track_counter.items() if k != "" and v > 1])
    if dup:
        raise RuntimeError(f"duplicate track_id found after merge, examples={dup[:20]}")

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in merged_rows:
            row = {c: str(r.get(c, "")) for c in cols}
            w.writerow(row)

    culture_counter = Counter(str(r.get("culture", "")).strip() for r in merged_rows)
    report = {
        "out_csv": str(out_path.resolve()),
        "n_rows": int(len(merged_rows)),
        "n_sources": int(len(in_paths)),
        "sources": sources,
        "n_cultures": int(len(culture_counter)),
        "culture_distribution": [{"culture": c, "count": int(v)} for c, v in sorted(culture_counter.items())],
    }
    rep_path = out_path.with_suffix(out_path.suffix + ".merge_report.json")
    with open(rep_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    report["report_path"] = str(rep_path.resolve())
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge multiple metadata.csv files into one.")
    ap.add_argument("--inputs", nargs="+", required=True, help="Input metadata.csv files")
    ap.add_argument("--out", required=True, help="Output merged metadata.csv")
    args = ap.parse_args()
    out = merge_metadata(inputs=args.inputs, out_csv=args.out)
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
