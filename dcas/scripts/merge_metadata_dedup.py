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


def merge_metadata_dedup(
    inputs: list[str | Path],
    out_csv: str | Path,
    require_audio_exists: bool = False,
) -> dict[str, Any]:
    in_paths = [Path(p) for p in inputs]
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    merged: list[dict[str, str]] = []
    all_fields: list[str] = []
    seen_fields: set[str] = set()
    sources: list[dict[str, Any]] = []
    seen_track_ids: set[str] = set()
    skipped_duplicates = 0
    skipped_missing_audio = 0

    for p in in_paths:
        metadata_dir = p.parent.absolute()
        rows, fields = _read_csv(p)
        required = {"track_id", "culture", "audio_path"}
        missing = sorted(list(required - set(fields)))
        if missing:
            raise RuntimeError(f"metadata missing required columns at {p}: {missing}")
        for c in fields:
            if c not in seen_fields:
                seen_fields.add(c)
                all_fields.append(c)
        n_before = len(merged)
        for r in rows:
            tid = str(r.get("track_id", "")).strip()
            if tid == "":
                continue
            if tid in seen_track_ids:
                skipped_duplicates += 1
                continue
            seen_track_ids.add(tid)
            rr = dict(r)
            rel = str(rr.get("audio_path", "")).strip()
            if rel == "":
                continue
            ap = Path(rel)
            if not ap.is_absolute():
                ap = metadata_dir / ap
            if require_audio_exists and not ap.exists():
                skipped_missing_audio += 1
                continue
            rr["audio_path"] = str(ap)
            merged.append(rr)
        sources.append(
            {
                "path": str(p.resolve()),
                "rows_before_dedup": len(rows),
                "rows_after_dedup": int(len(merged) - n_before),
            }
        )

    preferred = [
        "track_id",
        "culture",
        "audio_path",
        "source_dataset",
        "source_split",
        "source_index",
        "label",
        "affect_label",
    ]
    extra = [c for c in all_fields if c not in preferred]
    cols = [c for c in preferred if c in seen_fields] + extra

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in merged:
            w.writerow({c: str(r.get(c, "")) for c in cols})

    culture_counter = Counter(str(r.get("culture", "")).strip() for r in merged)
    report = {
        "out_csv": str(out_path.resolve()),
        "n_rows": int(len(merged)),
        "n_sources": int(len(in_paths)),
        "sources": sources,
        "n_cultures": int(len(culture_counter)),
        "culture_distribution": [{"culture": c, "count": int(v)} for c, v in sorted(culture_counter.items())],
        "skipped_duplicates": int(skipped_duplicates),
        "skipped_missing_audio": int(skipped_missing_audio),
    }
    rep_path = out_path.with_suffix(out_path.suffix + ".merge_report.json")
    with open(rep_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    report["report_path"] = str(rep_path.resolve())
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge multiple metadata.csv files and deduplicate rows by track_id.")
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--require_audio_exists", action="store_true")
    args = ap.parse_args()
    out = merge_metadata_dedup(
        inputs=args.inputs,
        out_csv=args.out,
        require_audio_exists=args.require_audio_exists,
    )
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
