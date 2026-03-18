from __future__ import annotations

import argparse
import csv
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dcas.embeddings import GeminiEmbedding2Config, GeminiEmbedding2Embedder
from dcas.serialization_json import write_json


def _require(row: dict[str, str], key: str) -> str:
    v = row.get(key)
    if v is None or str(v).strip() == "":
        raise ValueError(f"missing required column '{key}'")
    return str(v).strip()


def _load_cache(cache_file: Path) -> np.ndarray:
    return np.load(str(cache_file)).astype(np.float32)


def _resolve_api_key(api_key: str | None, api_key_file: str | Path | None) -> str | None:
    if api_key is not None and str(api_key).strip() != "":
        return str(api_key).strip()
    if api_key_file is None:
        return None
    p = Path(api_key_file)
    if not p.exists():
        raise RuntimeError(f"api_key_file does not exist: {p}")
    return p.read_text(encoding="utf-8").strip()


def build_tracks_with_gemini(
    metadata_csv: str | Path,
    out_npz: str | Path,
    model_id: str = "gemini-embedding-2-preview",
    api_key: str | None = None,
    api_key_file: str | Path | None = None,
    vertexai: bool = False,
    vertex_project: str | None = None,
    vertex_location: str | None = None,
    output_dimensionality: int = 768,
    task_type: str | None = None,
    max_seconds: float | None = 30.0,
    target_sample_rate: int = 16_000,
    window_count: int = 1,
    window_strategy: str = "single",
    window_aggregate: str = "mean",
    limit: int | None = None,
    skip_errors: bool = False,
    cache_dir: str | Path | None = None,
    dry_run: bool = False,
    max_workers: int = 1,
) -> dict[str, object]:
    metadata_path = Path(metadata_csv)
    out_path = Path(out_npz)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path = Path(cache_dir) if cache_dir is not None else out_path.with_suffix(out_path.suffix + ".cache")
    cache_path.mkdir(parents=True, exist_ok=True)

    cfg = GeminiEmbedding2Config(
        model_id=model_id,
        api_key=_resolve_api_key(api_key=api_key, api_key_file=api_key_file),
        vertexai=bool(vertexai),
        vertex_project=vertex_project,
        vertex_location=vertex_location,
        output_dimensionality=output_dimensionality,
        task_type=task_type,
        max_seconds=max_seconds,
        target_sample_rate=target_sample_rate,
        window_count=window_count,
        window_strategy=window_strategy,
        window_aggregate=window_aggregate,
    )

    worker_count = max(1, int(max_workers))
    thread_local = threading.local()

    prep_embedder: GeminiEmbedding2Embedder | None = None
    live_embedder: GeminiEmbedding2Embedder | None = None
    if dry_run:
        dry_cfg = GeminiEmbedding2Config(
            model_id=model_id,
            api_key="dry-run",
            output_dimensionality=output_dimensionality,
            task_type=task_type,
            max_seconds=max_seconds,
            target_sample_rate=target_sample_rate,
            window_count=window_count,
            window_strategy=window_strategy,
            window_aggregate=window_aggregate,
        )
        prep_embedder = GeminiEmbedding2Embedder.__new__(GeminiEmbedding2Embedder)
        prep_embedder.cfg = dry_cfg
        prep_embedder.session = None
    else:
        live_embedder = GeminiEmbedding2Embedder(cfg)

    with open(metadata_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames

    if not rows:
        raise RuntimeError("metadata is empty")

    required = {"track_id", "culture", "audio_path"}
    missing = [k for k in required if (fieldnames is None or k not in fieldnames)]
    if missing:
        raise RuntimeError(f"metadata missing required columns: {missing}")

    rows = sorted(rows, key=lambda r: (str(r.get("track_id", "")), str(r.get("audio_path", ""))))
    if limit is not None and int(limit) > 0:
        rows = rows[: int(limit)]

    total = len(rows)
    normalized_rows: list[dict[str, Any]] = []
    seen_track_ids: set[str] = set()
    duplicate_track_ids: list[str] = []

    for i, row in enumerate(rows, start=1):
        tid = _require(row, "track_id")
        if tid in seen_track_ids:
            duplicate_track_ids.append(tid)
        seen_track_ids.add(tid)

        cul = _require(row, "culture")
        rel_audio = _require(row, "audio_path")
        audio_path = Path(rel_audio)
        if not audio_path.is_absolute():
            audio_path = (metadata_path.parent / audio_path).resolve()

        normalized_rows.append(
            {
                "index": i,
                "track_id": tid,
                "culture": cul,
                "source_dataset": str(row.get("source_dataset", "")).strip(),
                "audio_path": audio_path,
                "title": str(row.get("title", "")).strip() or None,
                "raw_affect": str(row.get("affect_label", "")).strip(),
                "cache_file": cache_path / f"{tid}.npy",
                "cache_meta": cache_path / f"{tid}.json",
            }
        )

    if duplicate_track_ids:
        dup = sorted(set(duplicate_track_ids))
        raise RuntimeError(f"duplicate track_id found: {dup[:20]}")

    def _worker_embedder() -> GeminiEmbedding2Embedder:
        if dry_run:
            raise RuntimeError("dry_run does not use live embedder")
        cached = getattr(thread_local, "embedder", None)
        if cached is None:
            if live_embedder is not None and worker_count == 1:
                thread_local.embedder = live_embedder
            else:
                thread_local.embedder = GeminiEmbedding2Embedder(cfg)
        return thread_local.embedder

    def _process_row(item: dict[str, Any]) -> dict[str, Any]:
        tid = str(item["track_id"])
        cul = str(item["culture"])
        audio_path = Path(item["audio_path"])
        title = item["title"]
        cache_file = Path(item["cache_file"])
        cache_meta = Path(item["cache_meta"])

        if dry_run:
            assert prep_embedder is not None
            prep = prep_embedder.prepare_file_report(audio_path)
            prep["track_id"] = tid
            prep["culture"] = cul
            return {
                "index": int(item["index"]),
                "track_id": tid,
                "culture": cul,
                "source_dataset": str(item["source_dataset"]),
                "raw_affect": str(item["raw_affect"]),
                "prep": prep,
                "dry_run": True,
            }

        if cache_file.exists():
            prep = json.loads(cache_meta.read_text(encoding="utf-8")) if cache_meta.exists() else {}
            return {
                "index": int(item["index"]),
                "track_id": tid,
                "culture": cul,
                "source_dataset": str(item["source_dataset"]),
                "raw_affect": str(item["raw_affect"]),
                "prep": prep,
                "embedding": _load_cache(cache_file),
                "cache_hit": True,
            }

        emb, prep = _worker_embedder().embed_file(audio_path, title=title)
        np.save(str(cache_file), emb.astype(np.float32))
        cache_meta.write_text(json.dumps(prep, ensure_ascii=False, indent=2), encoding="utf-8")
        return {
            "index": int(item["index"]),
            "track_id": tid,
            "culture": cul,
            "source_dataset": str(item["source_dataset"]),
            "raw_affect": str(item["raw_affect"]),
            "prep": prep,
            "embedding": emb.astype(np.float32),
            "cache_hit": False,
        }

    results_by_index: dict[int, dict[str, Any]] = {}
    errors: list[str] = []
    cache_hits = 0

    if worker_count == 1:
        for item in normalized_rows:
            try:
                result = _process_row(item)
                results_by_index[int(result["index"])] = result
                if result.get("dry_run"):
                    prep = result["prep"]
                    print(
                        f"[{result['index']}/{total}] prepared: {result['track_id']} "
                        f"({result['culture']}) payload={prep['payload_bytes']} bytes"
                    )
                else:
                    source = "cache" if bool(result.get("cache_hit")) else "api"
                    print(
                        f"[{result['index']}/{total}] embedded: {result['track_id']} "
                        f"({result['culture']}) source={source}"
                    )
            except Exception as e:
                msg = f"row={item['index']}: {e}"
                if not skip_errors:
                    raise RuntimeError(msg) from e
                errors.append(msg)
                print(f"[{item['index']}/{total}] skipped: {msg}")
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as ex:
            future_map = {ex.submit(_process_row, item): item for item in normalized_rows}
            for fut in as_completed(future_map):
                item = future_map[fut]
                try:
                    result = fut.result()
                    results_by_index[int(result["index"])] = result
                    if result.get("dry_run"):
                        prep = result["prep"]
                        print(
                            f"[{result['index']}/{total}] prepared: {result['track_id']} "
                            f"({result['culture']}) payload={prep['payload_bytes']} bytes"
                        )
                    else:
                        source = "cache" if bool(result.get("cache_hit")) else "api"
                        print(
                            f"[{result['index']}/{total}] embedded: {result['track_id']} "
                            f"({result['culture']}) source={source}"
                        )
                except Exception as e:
                    msg = f"row={item['index']}: {e}"
                    if not skip_errors:
                        raise RuntimeError(msg) from e
                    errors.append(msg)
                    print(f"[{item['index']}/{total}] skipped: {msg}")

    track_ids: list[str] = []
    cultures: list[str] = []
    embeds: list[np.ndarray] = []
    affects: list[int] = []
    sources: list[str] = []
    has_source = True
    has_affect = True
    prep_reports: list[dict[str, Any]] = []

    for idx in sorted(results_by_index):
        result = results_by_index[idx]
        if result.get("dry_run"):
            prep_reports.append(result["prep"])
            continue

        track_ids.append(str(result["track_id"]))
        cultures.append(str(result["culture"]))
        embeds.append(np.asarray(result["embedding"], dtype=np.float32))
        cache_hits += 1 if bool(result.get("cache_hit")) else 0
        raw_source = str(result.get("source_dataset", "")).strip()
        if raw_source == "":
            has_source = False
        sources.append(raw_source)

        raw_affect = str(result.get("raw_affect", "")).strip()
        if raw_affect == "":
            has_affect = False
        affects.append(int(raw_affect) if raw_affect != "" else -1)

    manifest_path = out_path.with_suffix(out_path.suffix + ".manifest.json")

    if dry_run:
        manifest = {
            "metadata": str(metadata_path.resolve()),
            "out_tracks": None,
            "manifest_mode": "dry_run",
            "model_id": model_id,
            "output_dimensionality": output_dimensionality,
            "task_type": task_type,
            "vertexai": bool(vertexai),
            "vertex_project": vertex_project,
            "vertex_location": vertex_location,
            "max_seconds": max_seconds,
            "target_sample_rate": target_sample_rate,
            "window_count": int(window_count),
            "window_strategy": str(window_strategy),
            "window_aggregate": str(window_aggregate),
            "limit": limit,
            "skip_errors": bool(skip_errors),
            "max_workers": int(worker_count),
            "n_tracks_prepared": int(len(prep_reports)),
            "n_errors": int(len(errors)),
            "errors": errors,
            "prep_reports_preview": prep_reports[:20],
        }
        write_json(manifest_path, manifest)
        return {
            "out": None,
            "manifest": str(manifest_path),
            "n_tracks_prepared": int(len(prep_reports)),
            "errors": errors,
        }

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
    manifest = {
        "metadata": str(metadata_path.resolve()),
        "out_tracks": str(out_path.resolve()),
        "cache_dir": str(cache_path.resolve()),
        "model_id": model_id,
        "api_base": cfg.api_base,
        "vertexai": bool(cfg.vertexai),
        "vertex_project": cfg.vertex_project,
        "vertex_location": cfg.vertex_location,
        "output_dimensionality": output_dimensionality,
        "task_type": task_type,
        "max_seconds": max_seconds,
        "target_sample_rate": target_sample_rate,
        "window_count": int(window_count),
        "window_strategy": str(window_strategy),
        "window_aggregate": str(window_aggregate),
        "audio_mime_type": cfg.audio_mime_type,
        "limit": limit,
        "skip_errors": bool(skip_errors),
        "max_workers": int(worker_count),
        "n_tracks": int(emb_arr.shape[0]),
        "dim": int(emb_arr.shape[1]),
        "has_affect_label": bool(has_affect),
        "has_source_dataset": bool(has_source),
        "n_errors": int(len(errors)),
        "n_cache_hits": int(cache_hits),
        "errors": errors,
    }
    write_json(manifest_path, manifest)
    return {
        "out": str(out_path),
        "manifest": str(manifest_path),
        "n_tracks": int(emb_arr.shape[0]),
        "dim": int(emb_arr.shape[1]),
        "n_cache_hits": int(cache_hits),
        "errors": errors,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build tracks.npz from audio files using Gemini Embedding 2.",
    )
    ap.add_argument("--metadata", required=True, help="CSV with columns: track_id,culture,audio_path[,affect_label]")
    ap.add_argument("--out", required=True, help="Output tracks.npz path")
    ap.add_argument("--model_id", default="gemini-embedding-2-preview")
    ap.add_argument("--api_key", default=None, help="Optional Gemini API key; falls back to GEMINI_API_KEY")
    ap.add_argument("--api_key_file", default=None, help="Path to a local text file containing the Gemini API key")
    ap.add_argument("--vertexai", action="store_true", help="Use Vertex AI route instead of Gemini API REST route")
    ap.add_argument("--vertex_project", default=None)
    ap.add_argument("--vertex_location", default=None)
    ap.add_argument("--output_dimensionality", type=int, default=768)
    ap.add_argument("--task_type", default=None)
    ap.add_argument("--max_seconds", type=float, default=30.0)
    ap.add_argument("--target_sample_rate", type=int, default=16000)
    ap.add_argument("--window_count", type=int, default=1)
    ap.add_argument("--window_strategy", default="single")
    ap.add_argument("--window_aggregate", default="mean")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--skip_errors", action="store_true")
    ap.add_argument("--cache_dir", default=None)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--max_workers", type=int, default=1)
    args = ap.parse_args()

    out = build_tracks_with_gemini(
        metadata_csv=args.metadata,
        out_npz=args.out,
        model_id=args.model_id,
        api_key=args.api_key,
        api_key_file=args.api_key_file,
        vertexai=args.vertexai,
        vertex_project=args.vertex_project,
        vertex_location=args.vertex_location,
        output_dimensionality=args.output_dimensionality,
        task_type=args.task_type,
        max_seconds=args.max_seconds,
        target_sample_rate=args.target_sample_rate,
        window_count=int(args.window_count),
        window_strategy=str(args.window_strategy),
        window_aggregate=str(args.window_aggregate),
        limit=args.limit,
        skip_errors=args.skip_errors,
        cache_dir=args.cache_dir,
        dry_run=args.dry_run,
        max_workers=args.max_workers,
    )
    print(out)


if __name__ == "__main__":
    main()
