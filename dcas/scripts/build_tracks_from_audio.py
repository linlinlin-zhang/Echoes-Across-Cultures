from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from dcas.embeddings import CultureMERTConfig, CultureMERTEmbedder


def _require(row: dict[str, str], key: str) -> str:
    v = row.get(key)
    if v is None or str(v).strip() == "":
        raise ValueError(f"missing required column '{key}'")
    return str(v).strip()


def build_tracks_from_audio(
    metadata_csv: str | Path,
    out_npz: str | Path,
    model_id: str = "ntua-slp/CultureMERT-95M",
    device: str | None = None,
    max_seconds: float | None = 30.0,
    limit: int | None = None,
    skip_errors: bool = False,
) -> dict[str, object]:
    metadata_path = Path(metadata_csv)
    out_path = Path(out_npz)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = CultureMERTConfig(model_id=model_id, device=device, max_seconds=max_seconds)
    embedder = CultureMERTEmbedder(cfg)

    track_ids: list[str] = []
    cultures: list[str] = []
    embeds: list[np.ndarray] = []
    affects: list[int] = []
    has_affect = True
    errors: list[str] = []

    with open(metadata_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if limit is not None and int(limit) > 0:
        rows = rows[: int(limit)]

    total = len(rows)
    for i, row in enumerate(rows, start=1):
        try:
            tid = _require(row, "track_id")
            cul = _require(row, "culture")
            rel_audio = _require(row, "audio_path")
            audio_path = Path(rel_audio)
            if not audio_path.is_absolute():
                audio_path = (metadata_path.parent / audio_path).resolve()
            emb = embedder.embed_file(audio_path)

            track_ids.append(tid)
            cultures.append(cul)
            embeds.append(emb.astype(np.float32))

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

    if not embeds:
        raise RuntimeError("no embeddings generated")

    emb_arr = np.stack(embeds, axis=0).astype(np.float32)
    obj: dict[str, np.ndarray] = {
        "track_id": np.array(track_ids, dtype="<U128"),
        "culture": np.array(cultures, dtype="<U64"),
        "embedding": emb_arr,
    }
    if has_affect:
        obj["affect_label"] = np.array(affects, dtype=np.int64)

    np.savez_compressed(str(out_path), **obj)
    return {
        "out": str(out_path),
        "n_tracks": int(emb_arr.shape[0]),
        "dim": int(emb_arr.shape[1]),
        "errors": errors,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build tracks.npz from audio files using CultureMERT embeddings.",
    )
    ap.add_argument("--metadata", required=True, help="CSV with columns: track_id,culture,audio_path[,affect_label]")
    ap.add_argument("--out", required=True, help="Output tracks.npz path")
    ap.add_argument("--model_id", default="ntua-slp/CultureMERT-95M")
    ap.add_argument("--device", default=None, help="cpu/cuda, default auto")
    ap.add_argument("--max_seconds", type=float, default=30.0, help="Trim each track to this duration before embedding")
    ap.add_argument("--limit", type=int, default=None, help="Optional max number of rows")
    ap.add_argument("--skip_errors", action="store_true")
    args = ap.parse_args()

    out = build_tracks_from_audio(
        metadata_csv=args.metadata,
        out_npz=args.out,
        model_id=args.model_id,
        device=args.device,
        max_seconds=args.max_seconds,
        limit=args.limit,
        skip_errors=args.skip_errors,
    )
    print(out)


if __name__ == "__main__":
    main()

