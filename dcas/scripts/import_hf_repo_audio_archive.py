from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, hf_hub_download


def _slug(v: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(v).strip())
    return s.strip("_") or "item"


def _to_text(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, (list, dict)):
        return json.dumps(v, ensure_ascii=False)
    return str(v)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line == "":
                continue
            rows.append(json.loads(line))
    return rows


def import_hf_repo_audio_archive(
    repo_id: str,
    out_dir: str | Path,
    culture: str,
    limit: int | None = None,
    metadata_filename: str = "metadata.jsonl",
    file_field: str = "file",
    label_field: str = "type",
    extra_fields: list[str] | None = None,
    track_id_prefix: str | None = None,
    revision: str | None = None,
) -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    audio_out = out / "audio"
    audio_out.mkdir(parents=True, exist_ok=True)
    metadata_path = out / "metadata.csv"
    report_path = out / "import_report.json"

    prefix = _slug(track_id_prefix or repo_id.replace("/", "_"))
    extras = [x.strip() for x in (extra_fields or []) if x.strip()]

    local_metadata = Path(
        hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=metadata_filename,
            revision=revision,
        )
    )
    meta_rows = _read_jsonl(local_metadata)

    api = HfApi()
    repo_files = set(api.list_repo_files(repo_id=repo_id, repo_type="dataset", revision=revision))

    cols = [
        "track_id",
        "culture",
        "audio_path",
        "source_dataset",
        "source_split",
        "source_index",
        "label",
    ] + extras

    out_rows: list[dict[str, str]] = []
    skipped = 0
    errors: list[str] = []

    for idx, row in enumerate(meta_rows):
        if limit is not None and len(out_rows) >= int(limit):
            break
        try:
            remote_path = _to_text(row.get(file_field)).strip()
            if remote_path == "":
                raise RuntimeError(f"missing file field '{file_field}'")
            if remote_path not in repo_files:
                raise RuntimeError(f"file not found in repo listing: {remote_path}")

            ext = Path(remote_path).suffix.lower() or ".wav"
            tid = _slug(f"{prefix}_{idx:08d}")
            rel_path = Path("audio") / f"{tid}{ext}"
            abs_path = out / rel_path
            if not abs_path.exists():
                local_audio = Path(
                    hf_hub_download(
                        repo_id=repo_id,
                        repo_type="dataset",
                        filename=remote_path,
                        revision=revision,
                    )
                )
                shutil.copyfile(local_audio, abs_path)

            obj = {
                "track_id": tid,
                "culture": str(culture),
                "audio_path": str(rel_path.as_posix()),
                "source_dataset": str(repo_id),
                "source_split": "repo_archive",
                "source_index": str(idx),
                "label": _to_text(row.get(label_field)),
            }
            for extra in extras:
                obj[extra] = _to_text(row.get(extra))
            out_rows.append(obj)
        except Exception as e:
            skipped += 1
            errors.append(f"row={idx}: {e}")

    with open(metadata_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for row in out_rows:
            writer.writerow(row)

    report = {
        "repo_id": repo_id,
        "revision": revision,
        "metadata_filename": metadata_filename,
        "culture": culture,
        "limit": limit,
        "imported": len(out_rows),
        "skipped": skipped,
        "metadata_csv": str(metadata_path.resolve()),
        "audio_dir": str(audio_out.resolve()),
        "errors": errors[:200],
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Import audio files from an HF dataset repo backed by metadata.jsonl.")
    ap.add_argument("--repo_id", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--culture", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--metadata_filename", default="metadata.jsonl")
    ap.add_argument("--file_field", default="file")
    ap.add_argument("--label_field", default="type")
    ap.add_argument("--extra_fields", default="title,language,duration")
    ap.add_argument("--track_id_prefix", default=None)
    ap.add_argument("--revision", default=None)
    args = ap.parse_args()

    out = import_hf_repo_audio_archive(
        repo_id=str(args.repo_id),
        out_dir=str(args.out_dir),
        culture=str(args.culture),
        limit=args.limit,
        metadata_filename=str(args.metadata_filename),
        file_field=str(args.file_field),
        label_field=str(args.label_field),
        extra_fields=[x.strip() for x in str(args.extra_fields).split(",") if x.strip()],
        track_id_prefix=args.track_id_prefix,
        revision=args.revision,
    )
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
