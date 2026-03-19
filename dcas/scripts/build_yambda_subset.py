from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import fsspec
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


YAMBDA_DATASET = "yandex/yambda"
YAMBDA_EMBEDDINGS_URL = "https://huggingface.co/datasets/yandex/yambda/resolve/main/embeddings.parquet"


def _parse_row_groups(value: str) -> list[int]:
    out: list[int] = []
    for chunk in str(value).split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        out.append(int(chunk))
    if not out:
        raise ValueError("row_groups cannot be empty")
    return sorted(set(out))


def _large_list_to_matrix(arr: pa.Array) -> np.ndarray:
    if isinstance(arr, pa.ChunkedArray):
        arr = arr.combine_chunks()
    if not isinstance(arr, (pa.LargeListArray, pa.ListArray)):
        raise TypeError(f"expected list array, got: {type(arr)}")
    offsets = np.asarray(arr.offsets.to_numpy(), dtype=np.int64)
    if offsets.size <= 1:
        return np.zeros((0, 0), dtype=np.float32)
    dim = int(offsets[1] - offsets[0])
    flat = np.asarray(arr.values.to_numpy(zero_copy_only=False), dtype=np.float32)
    if dim <= 0:
        raise ValueError("embedding dimension must be positive")
    return flat.reshape(int(len(arr)), dim).astype(np.float32)


def _load_embedding_rows(
    row_groups: list[int],
    embedding_column: str,
) -> tuple[np.ndarray, np.ndarray]:
    fs = fsspec.filesystem("https")
    with fs.open(YAMBDA_EMBEDDINGS_URL, "rb") as f:
        pf = pq.ParquetFile(f)
        tables = [pf.read_row_group(int(rg), columns=["item_id", str(embedding_column)]) for rg in row_groups]
    table = pa.concat_tables(tables) if len(tables) > 1 else tables[0]
    item_ids = np.asarray(table.column("item_id").to_numpy(zero_copy_only=False), dtype=np.int64)
    emb = _large_list_to_matrix(table.column(str(embedding_column)))
    if item_ids.shape[0] != emb.shape[0]:
        raise RuntimeError("item_ids and embeddings have different lengths")
    return item_ids, emb


def build_yambda_subset(
    out_dir: str | Path,
    interaction_config: str = "flat-multievent-5b",
    row_groups: list[int] | None = None,
    embedding_column: str = "normalized_embed",
    max_events: int = 500_000,
    max_users: int = 80,
    min_interactions_per_user: int = 30,
    min_played_ratio: int = 50,
    organic_only: bool = False,
) -> dict[str, Any]:
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    selected_row_groups = row_groups or [0]
    embed_item_ids, embed_matrix = _load_embedding_rows(
        row_groups=selected_row_groups,
        embedding_column=str(embedding_column),
    )
    item_id_set = {int(x) for x in embed_item_ids.tolist()}
    item_to_embedding = {int(item_id): embed_matrix[idx] for idx, item_id in enumerate(embed_item_ids.tolist())}

    by_user: dict[str, list[dict[str, Any]]] = defaultdict(list)
    user_order: list[str] = []
    scanned_events = 0
    kept_events = 0
    ds = load_dataset(YAMBDA_DATASET, interaction_config, split="train", streaming=True)
    for row in ds:
        if scanned_events >= int(max_events):
            break
        scanned_events += 1
        if str(row.get("event_type", "")) != "listen":
            continue
        played_ratio = int(row.get("played_ratio_pct") or 0)
        if played_ratio < int(min_played_ratio):
            continue
        is_organic = int(row.get("is_organic") or 0)
        if bool(organic_only) and is_organic != 1:
            continue
        item_id = int(row["item_id"])
        if item_id not in item_id_set:
            continue

        uid = str(row["uid"])
        if uid not in by_user:
            user_order.append(uid)
        by_user[uid].append(
            {
                "timestamp": int(row.get("timestamp") or 0),
                "item_id": int(item_id),
                "played_ratio_pct": int(played_ratio),
                "track_length_seconds": int(row.get("track_length_seconds") or 0),
                "is_organic": int(is_organic),
            }
        )
        kept_events += 1

    eligible_users = [
        uid for uid in user_order if len(by_user.get(uid, [])) >= int(min_interactions_per_user)
    ][: int(max_users)]
    if not eligible_users:
        raise RuntimeError("no users satisfy the subset filters; try relaxing min_interactions_per_user or max_events")

    selected_item_ids: set[int] = set()
    selected_events: list[dict[str, Any]] = []
    for uid in eligible_users:
        rows = sorted(by_user[uid], key=lambda r: (int(r["timestamp"]), int(r["item_id"])))
        for row in rows:
            item_id = int(row["item_id"])
            selected_item_ids.add(item_id)
            selected_events.append(
                {
                    "user_id": f"yambda_u{uid}",
                    "track_id": f"yambda_{item_id}",
                    "weight": max(0.05, float(row["played_ratio_pct"]) / 100.0),
                    "timestamp": int(row["timestamp"]),
                    "played_ratio_pct": int(row["played_ratio_pct"]),
                    "track_length_seconds": int(row["track_length_seconds"]),
                    "is_organic": int(row["is_organic"]),
                    "raw_uid": str(uid),
                    "raw_item_id": int(item_id),
                }
            )

    ordered_item_ids = sorted(selected_item_ids)
    missing_embeddings = [item_id for item_id in ordered_item_ids if int(item_id) not in item_to_embedding]
    if missing_embeddings:
        raise RuntimeError(f"missing embeddings for {len(missing_embeddings)} selected items")

    track_ids = np.array([f"yambda_{item_id}" for item_id in ordered_item_ids], dtype="<U32")
    cultures = np.full((len(ordered_item_ids),), "global", dtype="<U16")
    sources = np.full((len(ordered_item_ids),), "yandex/yambda", dtype="<U32")
    embeddings = np.stack([item_to_embedding[int(item_id)] for item_id in ordered_item_ids], axis=0).astype(np.float32)

    tracks_path = out_root / "tracks.npz"
    np.savez_compressed(
        str(tracks_path),
        track_id=track_ids,
        culture=cultures,
        embedding=embeddings,
        source_dataset=sources,
    )

    metadata_path = out_root / "metadata.csv"
    with open(metadata_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "track_id",
                "culture",
                "source_dataset",
                "raw_item_id",
                "embedding_column",
                "embedding_row_groups",
            ],
        )
        writer.writeheader()
        for item_id in ordered_item_ids:
            writer.writerow(
                {
                    "track_id": f"yambda_{item_id}",
                    "culture": "global",
                    "source_dataset": "yandex/yambda",
                    "raw_item_id": int(item_id),
                    "embedding_column": str(embedding_column),
                    "embedding_row_groups": ",".join(str(x) for x in selected_row_groups),
                }
            )

    interactions_path = out_root / "interactions.csv"
    selected_events = sorted(selected_events, key=lambda r: (str(r["user_id"]), int(r["timestamp"]), str(r["track_id"])))
    with open(interactions_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "user_id",
                "track_id",
                "weight",
                "timestamp",
                "played_ratio_pct",
                "track_length_seconds",
                "is_organic",
                "raw_uid",
                "raw_item_id",
            ],
        )
        writer.writeheader()
        for row in selected_events:
            writer.writerow(row)

    report = {
        "dataset": YAMBDA_DATASET,
        "interaction_config": str(interaction_config),
        "embedding_url": YAMBDA_EMBEDDINGS_URL,
        "embedding_column": str(embedding_column),
        "embedding_row_groups": [int(x) for x in selected_row_groups],
        "max_events": int(max_events),
        "max_users": int(max_users),
        "min_interactions_per_user": int(min_interactions_per_user),
        "min_played_ratio": int(min_played_ratio),
        "organic_only": bool(organic_only),
        "scanned_events": int(scanned_events),
        "matched_events": int(kept_events),
        "n_users_selected": int(len(eligible_users)),
        "n_tracks_selected": int(len(ordered_item_ids)),
        "n_interactions_selected": int(len(selected_events)),
        "tracks_path": str(tracks_path.resolve()),
        "metadata_csv": str(metadata_path.resolve()),
        "interactions_csv": str(interactions_path.resolve()),
    }
    report_path = out_root / "subset_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    report["report_json"] = str(report_path.resolve())
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a small Yambda subset in repository-native tracks/interactions format.")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--interaction_config", default="flat-multievent-5b")
    ap.add_argument("--row_groups", default="0", help="Comma-separated row-group ids from embeddings.parquet")
    ap.add_argument("--embedding_column", default="normalized_embed", choices=["embed", "normalized_embed"])
    ap.add_argument("--max_events", type=int, default=500000)
    ap.add_argument("--max_users", type=int, default=80)
    ap.add_argument("--min_interactions_per_user", type=int, default=30)
    ap.add_argument("--min_played_ratio", type=int, default=50)
    ap.add_argument("--organic_only", action="store_true")
    args = ap.parse_args()

    rep = build_yambda_subset(
        out_dir=str(args.out_dir),
        interaction_config=str(args.interaction_config),
        row_groups=_parse_row_groups(str(args.row_groups)),
        embedding_column=str(args.embedding_column),
        max_events=int(args.max_events),
        max_users=int(args.max_users),
        min_interactions_per_user=int(args.min_interactions_per_user),
        min_played_ratio=int(args.min_played_ratio),
        organic_only=bool(args.organic_only),
    )
    print(json.dumps(rep, ensure_ascii=False))


if __name__ == "__main__":
    main()
