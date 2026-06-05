from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from dcas.data.npz_tracks import load_tracks


def _read_csv(path: str | Path) -> tuple[list[dict[str, str]], list[str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        return rows, list(reader.fieldnames or [])


def _write_csv(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def align_assets_to_tracks(
    tracks_path: str | Path,
    metadata_in: str | Path,
    metadata_out: str | Path,
    interactions_in: str | Path | None = None,
    interactions_out: str | Path | None = None,
    constraints_in: str | Path | None = None,
    constraints_out: str | Path | None = None,
) -> dict[str, Any]:
    tracks = load_tracks(str(tracks_path))
    track_ids = {str(x) for x in tracks.track_id.tolist()}

    meta_rows, meta_fields = _read_csv(metadata_in)
    kept_meta = [row for row in meta_rows if str(row.get("track_id", "")).strip() in track_ids]
    _write_csv(metadata_out, kept_meta, meta_fields)

    report: dict[str, Any] = {
        "tracks": str(Path(tracks_path).resolve()),
        "track_count": int(len(track_ids)),
        "metadata_in": str(Path(metadata_in).resolve()),
        "metadata_out": str(Path(metadata_out).resolve()),
        "metadata_rows_in": int(len(meta_rows)),
        "metadata_rows_out": int(len(kept_meta)),
        "metadata_rows_dropped": int(len(meta_rows) - len(kept_meta)),
    }

    if interactions_in and interactions_out:
        inter_rows, inter_fields = _read_csv(interactions_in)
        kept_inter = [row for row in inter_rows if str(row.get("track_id", "")).strip() in track_ids]
        _write_csv(interactions_out, kept_inter, inter_fields)
        report.update(
            {
                "interactions_in": str(Path(interactions_in).resolve()),
                "interactions_out": str(Path(interactions_out).resolve()),
                "interactions_rows_in": int(len(inter_rows)),
                "interactions_rows_out": int(len(kept_inter)),
                "interactions_rows_dropped": int(len(inter_rows) - len(kept_inter)),
            }
        )

    if constraints_in and constraints_out:
        constraint_rows = _read_jsonl(constraints_in)
        kept_constraints = [
            row
            for row in constraint_rows
            if str(row.get("track_id_a", "")).strip() in track_ids
            and str(row.get("track_id_b", "")).strip() in track_ids
        ]
        _write_jsonl(constraints_out, kept_constraints)
        report.update(
            {
                "constraints_in": str(Path(constraints_in).resolve()),
                "constraints_out": str(Path(constraints_out).resolve()),
                "constraints_rows_in": int(len(constraint_rows)),
                "constraints_rows_out": int(len(kept_constraints)),
                "constraints_rows_dropped": int(len(constraint_rows) - len(kept_constraints)),
            }
        )

    report_path = Path(metadata_out).with_suffix(Path(metadata_out).suffix + ".align_report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    report["report_json"] = str(report_path.resolve())
    return report


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Filter metadata/interactions/constraints down to the track_ids present in a tracks.npz file."
    )
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--metadata_in", required=True)
    ap.add_argument("--metadata_out", required=True)
    ap.add_argument("--interactions_in", default=None)
    ap.add_argument("--interactions_out", default=None)
    ap.add_argument("--constraints_in", default=None)
    ap.add_argument("--constraints_out", default=None)
    args = ap.parse_args()

    rep = align_assets_to_tracks(
        tracks_path=str(args.tracks),
        metadata_in=str(args.metadata_in),
        metadata_out=str(args.metadata_out),
        interactions_in=str(args.interactions_in) if args.interactions_in else None,
        interactions_out=str(args.interactions_out) if args.interactions_out else None,
        constraints_in=str(args.constraints_in) if args.constraints_in else None,
        constraints_out=str(args.constraints_out) if args.constraints_out else None,
    )
    print(json.dumps(rep, ensure_ascii=False))


if __name__ == "__main__":
    main()
