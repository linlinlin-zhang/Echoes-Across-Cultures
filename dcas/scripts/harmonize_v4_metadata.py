from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torchaudio

from dcas.scripts.harmonize_v3_metadata import (
    DEFAULT_FMA_METADATA_ZIP,
    harmonize_metadata,
)


REQUIRED_FIELDS = [
    "track_id",
    "culture",
    "audio_path",
    "source_dataset",
    "source_split",
    "source_index",
    "duration_sec",
    "sample_rate",
    "channels",
    "era",
    "region",
]

RECOMMENDED_FIELDS = [
    "fine_label",
    "label",
    "substyle",
    "instrument",
    "instrument_family",
    "language",
    "title",
    "artist",
    "license",
    "license_note",
    "url",
    "recording_condition",
    "notes",
]

GOVERNANCE_FIELDS = [
    "schema_version",
    "dataset_version",
    "import_batch",
    "dedup_group_id",
    "dedup_keep",
    "qc_status",
    "qc_notes",
    "embedding_status_culturemert",
    "embedding_status_gemini",
    "drop_reason",
]

FINAL_FIELD_ORDER = REQUIRED_FIELDS + ["coarse_label", "is_instrumental"] + RECOMMENDED_FIELDS + GOVERNANCE_FIELDS


def _read_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader), list(reader.fieldnames or [])


def _write_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none", "null"}:
        return ""
    return text


def _normalize_row(
    row: dict[str, str],
    metadata_dir: Path,
    row_index: int,
    dataset_version: str,
    schema_version: str,
    import_batch: str | None,
) -> dict[str, str]:
    out = {key: _clean_text(row.get(key, "")) for key in set(FINAL_FIELD_ORDER).union(row.keys())}
    out["track_id"] = _clean_text(row.get("track_id", ""))
    out["culture"] = _clean_text(row.get("culture", ""))
    out["audio_path"] = _clean_text(row.get("audio_path", ""))
    out["source_dataset"] = _clean_text(row.get("source_dataset", ""))
    out["source_split"] = _clean_text(row.get("source_split", "")) or "unknown"
    out["source_index"] = _clean_text(row.get("source_index", "")) or str(row_index)
    out["duration_sec"] = _clean_text(row.get("duration_sec", ""))
    out["sample_rate"] = _clean_text(row.get("sample_rate", "")) or _clean_text(row.get("sample_rate_hz", ""))
    out["channels"] = _clean_text(row.get("channels", "")) or _clean_text(row.get("num_channels", ""))
    out["fine_label"] = _clean_text(row.get("fine_label", "")) or _clean_text(row.get("label", ""))
    out["label"] = _clean_text(row.get("label", ""))
    out["substyle"] = _clean_text(row.get("substyle", ""))
    out["instrument"] = _clean_text(row.get("instrument", ""))
    out["instrument_family"] = _clean_text(row.get("instrument_family", ""))
    out["language"] = _clean_text(row.get("language", "")).lower()
    out["title"] = _clean_text(row.get("title", ""))
    out["artist"] = _clean_text(row.get("artist", ""))
    out["license"] = _clean_text(row.get("license", ""))
    out["license_note"] = _clean_text(row.get("license_note", ""))
    out["url"] = _clean_text(row.get("url", ""))
    out["recording_condition"] = _clean_text(row.get("recording_condition", ""))
    out["notes"] = _clean_text(row.get("notes", ""))
    out["era"] = _clean_text(row.get("era", ""))
    out["region"] = _clean_text(row.get("region", "")) or out["culture"]

    out["schema_version"] = schema_version
    out["dataset_version"] = dataset_version
    out["import_batch"] = _clean_text(import_batch) or dataset_version
    out["dedup_group_id"] = out["track_id"]
    out["dedup_keep"] = "1"
    out["qc_status"] = _clean_text(row.get("qc_status", "")) or "pending"
    out["qc_notes"] = _clean_text(row.get("qc_notes", ""))
    out["embedding_status_culturemert"] = _clean_text(row.get("embedding_status_culturemert", "")) or "pending"
    out["embedding_status_gemini"] = _clean_text(row.get("embedding_status_gemini", "")) or "pending"
    out["drop_reason"] = _clean_text(row.get("drop_reason", ""))

    if out["audio_path"] and (out["duration_sec"] == "" or out["sample_rate"] == "" or out["channels"] == ""):
        audio_path = Path(out["audio_path"])
        if not audio_path.is_absolute():
            audio_path = (metadata_dir / audio_path).resolve()
        try:
            info = torchaudio.info(str(audio_path))
            if out["duration_sec"] == "" and int(info.sample_rate) > 0:
                out["duration_sec"] = str(float(info.num_frames) / float(info.sample_rate))
            if out["sample_rate"] == "":
                out["sample_rate"] = str(int(info.sample_rate))
            if out["channels"] == "":
                out["channels"] = str(int(info.num_channels))
        except Exception:
            pass
    return out


def harmonize_v4_metadata(
    metadata_csv: str | Path,
    out_clean_csv: str | Path,
    out_harmonized_csv: str | Path,
    dataset_version: str,
    schema_version: str = "v4.0",
    import_batch: str | None = None,
    fma_metadata_zip: str | Path | None = None,
) -> dict[str, Any]:
    in_path = Path(metadata_csv)
    clean_path = Path(out_clean_csv)
    harmonized_path = Path(out_harmonized_csv)

    rows, fieldnames = _read_rows(in_path)
    normalized = [
        _normalize_row(
            row=row,
            metadata_dir=in_path.parent,
            row_index=index,
            dataset_version=dataset_version,
            schema_version=schema_version,
            import_batch=import_batch,
        )
        for index, row in enumerate(rows)
    ]
    clean_fields = list(dict.fromkeys(FINAL_FIELD_ORDER + fieldnames))
    _write_rows(clean_path, normalized, clean_fields)

    harmonize_metadata(
        metadata_csv=clean_path,
        out_csv=harmonized_path,
        fma_metadata_zip=fma_metadata_zip or DEFAULT_FMA_METADATA_ZIP,
    )
    harmonized_rows, harmonized_fields = _read_rows(harmonized_path)
    final_fields = list(dict.fromkeys(FINAL_FIELD_ORDER + harmonized_fields))
    _write_rows(harmonized_path, harmonized_rows, final_fields)

    report = {
        "metadata_in": str(in_path.resolve()),
        "metadata_clean": str(clean_path.resolve()),
        "metadata_harmonized": str(harmonized_path.resolve()),
        "dataset_version": dataset_version,
        "schema_version": schema_version,
        "n_rows": int(len(harmonized_rows)),
        "clean_field_count": int(len(clean_fields)),
        "harmonized_field_count": int(len(final_fields)),
    }
    report_path = harmonized_path.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    report["report_json"] = str(report_path.resolve())
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Normalize V4 metadata schema and add harmonized label fields.")
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--out_clean", required=True)
    ap.add_argument("--out_harmonized", required=True)
    ap.add_argument("--dataset_version", required=True)
    ap.add_argument("--schema_version", default="v4.0")
    ap.add_argument("--import_batch", default=None)
    ap.add_argument("--fma_metadata_zip", default=str(DEFAULT_FMA_METADATA_ZIP))
    args = ap.parse_args()

    report = harmonize_v4_metadata(
        metadata_csv=args.metadata,
        out_clean_csv=args.out_clean,
        out_harmonized_csv=args.out_harmonized,
        dataset_version=args.dataset_version,
        schema_version=args.schema_version,
        import_batch=args.import_batch,
        fma_metadata_zip=args.fma_metadata_zip,
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
