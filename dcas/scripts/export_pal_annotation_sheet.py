from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _load_metadata(metadata_csv: str) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    with open(metadata_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tid = str(row.get("track_id", "")).strip()
            if tid:
                rows[tid] = {str(k): str(v) for k, v in row.items()}
    return rows


def export_pal_annotation_sheet(tasks_path: str, metadata_csv: str, out_csv: str) -> dict[str, object]:
    meta = _load_metadata(metadata_csv)
    out_rows: list[dict[str, object]] = []
    with open(tasks_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            a = str(obj.get("track_id", "")).strip()
            b = str(obj.get("compare_to", "")).strip()
            if not a or not b:
                continue
            ma = meta.get(a, {})
            mb = meta.get(b, {})
            out_rows.append(
                {
                    "task_id": f"pal_{i:04d}",
                    "track_id_a": a,
                    "track_id_b": b,
                    "culture_a": ma.get("culture", ""),
                    "culture_b": mb.get("culture", ""),
                    "label_a": ma.get("label", ""),
                    "label_b": mb.get("label", ""),
                    "title_a": ma.get("title", ""),
                    "title_b": mb.get("title", ""),
                    "audio_path_a": ma.get("audio_path", ""),
                    "audio_path_b": mb.get("audio_path", ""),
                    "source_url_a": ma.get("source_url", ""),
                    "source_url_b": mb.get("source_url", ""),
                    "uncertainty": obj.get("uncertainty", ""),
                    "uncertainty_method": obj.get("uncertainty_method", ""),
                    "question": obj.get("question", ""),
                    "similar": "",
                    "rationale": "",
                    "annotator": "",
                    "notes": "",
                }
            )

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()) if out_rows else [])
        if out_rows:
            writer.writeheader()
            writer.writerows(out_rows)

    return {
        "tasks_path": str(tasks_path),
        "metadata_csv": str(metadata_csv),
        "out_csv": str(out_path),
        "n_tasks": int(len(out_rows)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Export PAL tasks to an annotation-friendly CSV sheet.")
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rep = export_pal_annotation_sheet(tasks_path=str(args.tasks), metadata_csv=str(args.metadata), out_csv=str(args.out))
    print(json.dumps(rep, ensure_ascii=False))


if __name__ == "__main__":
    main()
