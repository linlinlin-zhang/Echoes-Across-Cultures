from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

try:
    from datasets import Audio, load_dataset
except Exception:  # pragma: no cover - optional runtime dependency
    Audio = None
    load_dataset = None


def _slug(v: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(v).strip())
    return s.strip("_") or "item"


def _to_text(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, (list, dict)):
        return json.dumps(v, ensure_ascii=False)
    return str(v)


def _load_json(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    p = Path(path)
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_culture(
    row: dict[str, Any],
    mode: str,
    default_value: str,
    culture_column: str | None,
    culture_map: dict[str, Any],
) -> str:
    if mode == "constant":
        return str(default_value)
    if culture_column is None:
        raise ValueError("culture_column is required when culture_mode is not 'constant'")
    raw = row.get(culture_column)
    key = _to_text(raw).strip()
    if mode == "column":
        if key == "":
            return str(default_value)
        return key
    if mode == "map":
        if key in culture_map:
            return str(culture_map[key])
        key_lower = key.lower()
        if key_lower in culture_map:
            return str(culture_map[key_lower])
        return str(default_value)
    raise ValueError(f"unknown culture mode: {mode}")


def import_hf_audio_dataset(
    dataset: str,
    split: str,
    out_dir: str | Path,
    config: str | None = None,
    audio_column: str = "audio",
    track_id_prefix: str | None = None,
    track_id_column: str | None = None,
    limit: int | None = None,
    streaming: bool = False,
    culture_mode: str = "constant",
    culture_value: str = "west",
    culture_column: str | None = None,
    culture_map_json: str | Path | None = None,
    label_column: str | None = "genre",
    affect_column: str | None = None,
    extra_columns: list[str] | None = None,
) -> dict[str, Any]:
    if load_dataset is None or Audio is None:
        raise ImportError("datasets is required. Please install: pip install datasets")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    audio_out = out / "audio"
    audio_out.mkdir(parents=True, exist_ok=True)
    metadata_path = out / "metadata.csv"
    report_path = out / "import_report.json"

    prefix = _slug(track_id_prefix or dataset.replace("/", "_"))
    culture_map = _load_json(culture_map_json)
    extra_cols = [c.strip() for c in (extra_columns or []) if c.strip()]

    ds = load_dataset(path=dataset, name=config, split=split, streaming=bool(streaming))
    ds = ds.cast_column(audio_column, Audio(decode=False))

    required_cols = ["track_id", "culture", "audio_path"]
    optional_cols = ["source_dataset", "source_split", "source_index"]
    if label_column:
        optional_cols.append("label")
    if affect_column:
        optional_cols.append("affect_label")
    optional_cols.extend(extra_cols)
    cols = required_cols + optional_cols

    rows: list[dict[str, str]] = []
    errors: list[str] = []
    used_track_ids: set[str] = set()
    imported = 0
    skipped = 0

    for idx, row in enumerate(ds):
        if limit is not None and imported >= int(limit):
            break
        try:
            if track_id_column:
                raw_tid = _to_text(row.get(track_id_column, "")).strip()
                if raw_tid == "":
                    raw_tid = f"{prefix}_{idx:08d}"
            else:
                raw_tid = f"{prefix}_{idx:08d}"
            tid = _slug(raw_tid)
            if tid in used_track_ids:
                tid = f"{tid}_{idx:08d}"
            used_track_ids.add(tid)

            audio_obj = row.get(audio_column)
            if not isinstance(audio_obj, dict):
                raise RuntimeError(f"audio column '{audio_column}' is not a dict at row {idx}")

            audio_bytes = audio_obj.get("bytes")
            source_path = _to_text(audio_obj.get("path", "")).strip()
            ext = Path(source_path).suffix.lower() if source_path else ""
            if ext == "":
                ext = ".wav"
            rel_path = Path("audio") / f"{tid}{ext}"
            abs_path = out / rel_path
            abs_path.parent.mkdir(parents=True, exist_ok=True)

            if audio_bytes is not None:
                if isinstance(audio_bytes, memoryview):
                    audio_bytes = audio_bytes.tobytes()
                if not isinstance(audio_bytes, (bytes, bytearray)):
                    raise RuntimeError(f"unsupported audio bytes type at row {idx}: {type(audio_bytes)}")
                with open(abs_path, "wb") as f:
                    f.write(bytes(audio_bytes))
            else:
                src = Path(source_path)
                if not src.exists():
                    raise RuntimeError(f"audio bytes/path unavailable for row {idx}")
                abs_path.write_bytes(src.read_bytes())

            culture = _resolve_culture(
                row=row,
                mode=culture_mode,
                default_value=culture_value,
                culture_column=culture_column,
                culture_map=culture_map,
            )

            obj: dict[str, str] = {
                "track_id": tid,
                "culture": culture,
                "audio_path": str(rel_path.as_posix()),
                "source_dataset": str(dataset),
                "source_split": str(split),
                "source_index": str(idx),
            }
            if label_column:
                obj["label"] = _to_text(row.get(label_column))
            if affect_column:
                obj["affect_label"] = _to_text(row.get(affect_column))
            for c in extra_cols:
                obj[c] = _to_text(row.get(c))

            rows.append(obj)
            imported += 1
            if imported % 50 == 0:
                print(f"imported={imported}")
        except Exception as e:
            skipped += 1
            errors.append(f"row={idx}: {e}")

    with open(metadata_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    report = {
        "dataset": dataset,
        "config": config,
        "split": split,
        "streaming": bool(streaming),
        "audio_column": audio_column,
        "track_id_prefix": prefix,
        "track_id_column": track_id_column,
        "culture_mode": culture_mode,
        "culture_value": culture_value,
        "culture_column": culture_column,
        "label_column": label_column,
        "affect_column": affect_column,
        "extra_columns": extra_cols,
        "limit": limit,
        "imported": int(imported),
        "skipped": int(skipped),
        "metadata_csv": str(metadata_path.resolve()),
        "audio_dir": str(audio_out.resolve()),
        "errors": errors[:200],
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    return report


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Import a HuggingFace audio dataset and emit standard metadata.csv for DCAS.",
    )
    ap.add_argument("--dataset", required=True, help="HF dataset id, e.g. sanchit-gandhi/gtzan")
    ap.add_argument("--split", default="train")
    ap.add_argument("--config", default=None)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--audio_column", default="audio")
    ap.add_argument("--track_id_prefix", default=None)
    ap.add_argument("--track_id_column", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--streaming", action="store_true")

    ap.add_argument("--culture_mode", choices=["constant", "column", "map"], default="constant")
    ap.add_argument("--culture_value", default="west")
    ap.add_argument("--culture_column", default=None)
    ap.add_argument("--culture_map_json", default=None, help="JSON mapping raw label -> culture")

    ap.add_argument("--label_column", default="genre")
    ap.add_argument("--affect_column", default=None)
    ap.add_argument(
        "--extra_columns",
        default="",
        help="Comma-separated extra columns copied from HF row into metadata.",
    )
    args = ap.parse_args()

    extra_cols = [x.strip() for x in str(args.extra_columns).split(",") if x.strip()]
    out = import_hf_audio_dataset(
        dataset=args.dataset,
        split=args.split,
        out_dir=args.out_dir,
        config=args.config,
        audio_column=args.audio_column,
        track_id_prefix=args.track_id_prefix,
        track_id_column=args.track_id_column,
        limit=args.limit,
        streaming=bool(args.streaming),
        culture_mode=args.culture_mode,
        culture_value=args.culture_value,
        culture_column=args.culture_column,
        culture_map_json=args.culture_map_json,
        label_column=args.label_column,
        affect_column=args.affect_column,
        extra_columns=extra_cols,
    )
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
