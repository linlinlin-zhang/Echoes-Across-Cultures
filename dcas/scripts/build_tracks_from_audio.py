from __future__ import annotations

import argparse
import csv
import os
import re
from pathlib import Path

import numpy as np

from dcas.embeddings import CultureMERTConfig, CultureMERTEmbedder
from dcas.serialization_json import write_json


def _require(row: dict[str, str], key: str) -> str:
    v = row.get(key)
    if v is None or str(v).strip() == "":
        raise ValueError(f"missing required column '{key}'")
    return str(v).strip()


def _resolve_audio_path(rel_audio: str, metadata_path: Path) -> Path:
    raw = str(rel_audio).strip()
    if os.name == "nt":
        normalized = raw.replace("\\", "/")
        m = re.match(r"^/mnt/([a-zA-Z])/(.+)$", normalized)
        if m:
            return Path(f"{m.group(1).upper()}:/{m.group(2)}")

    audio_path = Path(raw)
    if not audio_path.is_absolute():
        audio_path = (metadata_path.parent / audio_path).resolve()
    return audio_path


def build_tracks_from_audio(
    metadata_csv: str | Path,
    out_npz: str | Path,
    model_id: str = "ntua-slp/CultureMERT-95M",
    device: str | None = None,
    pooling: str = "mean",
    layer_indices: list[int] | None = None,
    layer_weights: list[float] | None = None,
    max_seconds: float | None = 30.0,
    window_count: int = 1,
    window_strategy: str = "single",
    window_aggregate: str = "mean",
    limit: int | None = None,
    skip_errors: bool = False,
) -> dict[str, object]:
    metadata_path = Path(metadata_csv)
    out_path = Path(out_npz)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = CultureMERTConfig(
        model_id=model_id,
        device=device,
        pooling=pooling,
        layer_indices=layer_indices,
        layer_weights=layer_weights,
        max_seconds=max_seconds,
        window_count=window_count,
        window_strategy=window_strategy,
        window_aggregate=window_aggregate,
    )
    embedder = CultureMERTEmbedder(cfg)

    track_ids: list[str] = []
    cultures: list[str] = []
    embeds: list[np.ndarray] = []
    affects: list[int] = []
    sources: list[str] = []
    has_affect = True
    has_source = True
    errors: list[str] = []

    with open(metadata_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise RuntimeError("metadata is empty")

    required = {"track_id", "culture", "audio_path"}
    missing = [k for k in required if (reader.fieldnames is None or k not in reader.fieldnames)]
    if missing:
        raise RuntimeError(f"metadata missing required columns: {missing}")

    rows = sorted(
        rows,
        key=lambda r: (
            str(r.get("track_id", "")),
            str(r.get("audio_path", "")),
        ),
    )

    if limit is not None and int(limit) > 0:
        rows = rows[: int(limit)]

    total = len(rows)
    seen_track_ids: set[str] = set()
    duplicate_track_ids: list[str] = []
    for i, row in enumerate(rows, start=1):
        try:
            tid = _require(row, "track_id")
            if tid in seen_track_ids:
                duplicate_track_ids.append(tid)
            seen_track_ids.add(tid)
            cul = _require(row, "culture")
            rel_audio = _require(row, "audio_path")
            audio_path = _resolve_audio_path(rel_audio, metadata_path)
            emb = embedder.embed_file(audio_path)

            track_ids.append(tid)
            cultures.append(cul)
            embeds.append(emb.astype(np.float32))
            raw_source = str(row.get("source_dataset", "")).strip()
            if raw_source == "":
                has_source = False
            sources.append(raw_source)

            raw_affect = str(row.get("affect_label", "")).strip()
            if raw_affect == "":
                has_affect = False
            affects.append(int(raw_affect) if raw_affect != "" else -1)
            print(f"[{i}/{total}] embedded: {tid} ({cul})")
        except Exception as e:
            msg = f"row={i}: {e}"
            if not skip_errors:
                raise RuntimeError(msg) from e
            errors.append(msg)
            print(f"[{i}/{total}] skipped: {msg}")

    if duplicate_track_ids:
        dup = sorted(set(duplicate_track_ids))
        raise RuntimeError(f"duplicate track_id found: {dup[:20]}")

    if not embeds:
        raise RuntimeError("no embeddings generated")

    emb_arr = np.stack(embeds, axis=0).astype(np.float32)
    obj: dict[str, np.ndarray] = {
        "track_id": np.array(track_ids, dtype="<U128"),
        "culture": np.array(cultures, dtype="<U64"),
        "embedding": emb_arr,
    }
    if has_source:
        obj["source_dataset"] = np.array(sources, dtype="<U128")
    if has_affect:
        obj["affect_label"] = np.array(affects, dtype=np.int64)

    np.savez_compressed(str(out_path), **obj)
    manifest_path = out_path.with_suffix(out_path.suffix + ".manifest.json")
    manifest = {
        "metadata": str(metadata_path.resolve()),
        "out_tracks": str(out_path.resolve()),
        "model_id": model_id,
        "pooling": pooling,
        "layer_indices": layer_indices,
        "layer_weights": layer_weights,
        "device": device or "auto",
        "max_seconds": max_seconds,
        "window_count": int(window_count),
        "window_strategy": str(window_strategy),
        "window_aggregate": str(window_aggregate),
        "limit": limit,
        "skip_errors": bool(skip_errors),
        "n_tracks": int(emb_arr.shape[0]),
        "dim": int(emb_arr.shape[1]),
        "has_affect_label": bool(has_affect),
        "has_source_dataset": bool(has_source),
        "n_errors": int(len(errors)),
        "errors": errors,
    }
    write_json(manifest_path, manifest)
    return {
        "out": str(out_path),
        "manifest": str(manifest_path),
        "n_tracks": int(emb_arr.shape[0]),
        "dim": int(emb_arr.shape[1]),
        "errors": errors,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build tracks.npz from audio files using CultureMERT embeddings.",
    )
    ap.add_argument(
        "--metadata",
        required=True,
        help="CSV with columns: track_id,culture,audio_path[,affect_label]",
    )
    ap.add_argument("--out", required=True, help="Output tracks.npz path")
    ap.add_argument("--model_id", default="ntua-slp/CultureMERT-95M")
    ap.add_argument("--device", default=None, help="cpu/cuda, default auto")
    ap.add_argument(
        "--pooling",
        default="mean",
        choices=["mean", "cls"],
        help="Embedding pooling strategy",
    )
    ap.add_argument(
        "--layer_indices",
        nargs="*",
        type=int,
        default=None,
        help="Optional layer indices to aggregate",
    )
    ap.add_argument(
        "--layer_weights",
        nargs="*",
        type=float,
        default=None,
        help="Optional normalized or unnormalized weights for selected layers",
    )
    ap.add_argument(
        "--max_seconds",
        type=float,
        default=30.0,
        help="Trim each track to this duration before embedding",
    )
    ap.add_argument(
        "--window_count",
        type=int,
        default=1,
        help="Number of windows to sample and aggregate per track",
    )
    ap.add_argument(
        "--window_strategy",
        default="single",
        help="Window sampling strategy: single or uniform",
    )
    ap.add_argument("--window_aggregate", default="mean", help="Window aggregation strategy")
    ap.add_argument("--limit", type=int, default=None, help="Optional max number of rows")
    ap.add_argument("--skip_errors", action="store_true")
    args = ap.parse_args()

    out = build_tracks_from_audio(
        metadata_csv=args.metadata,
        out_npz=args.out,
        model_id=args.model_id,
        device=args.device,
        pooling=args.pooling,
        layer_indices=args.layer_indices,
        layer_weights=args.layer_weights,
        max_seconds=args.max_seconds,
        window_count=int(args.window_count),
        window_strategy=str(args.window_strategy),
        window_aggregate=str(args.window_aggregate),
        limit=args.limit,
        skip_errors=args.skip_errors,
    )
    print(out)


if __name__ == "__main__":
    main()
