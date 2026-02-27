from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def synthesize_interactions(
    metadata_csv: str | Path,
    out_csv: str | Path,
    users_per_culture: int = 20,
    tracks_per_user: int = 50,
    min_weight: float = 0.5,
    max_weight: float = 2.0,
    genre_column: str = "label",
    seed: int = 42,
) -> dict[str, int | str]:
    meta_path = Path(metadata_csv)
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(meta_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fields = set(reader.fieldnames or [])

    required = {"track_id", "culture"}
    missing = sorted(list(required - fields))
    if missing:
        raise RuntimeError(f"metadata missing required columns: {missing}")
    has_genre = genre_column in fields

    by_culture: dict[str, list[dict[str, str]]] = defaultdict(list)
    by_culture_genre: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for r in rows:
        culture = str(r.get("culture", "")).strip()
        if culture == "":
            continue
        by_culture[culture].append(r)
        if has_genre:
            g = str(r.get(genre_column, "")).strip()
            if g != "":
                by_culture_genre[(culture, g)].append(r)

    rng = np.random.default_rng(int(seed))
    n_rows = 0
    n_users = 0
    seen_pairs: set[tuple[str, str]] = set()

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["user_id", "track_id", "weight"])
        writer.writeheader()

        for culture in sorted(by_culture.keys()):
            pool = by_culture[culture]
            if not pool:
                continue
            genres = sorted({str(x.get(genre_column, "")).strip() for x in pool if str(x.get(genre_column, "")).strip() != ""})
            for i in range(int(users_per_culture)):
                uid = f"{culture}_u{i:03d}"
                n_users += 1
                preferred_genre = None
                if has_genre and genres:
                    preferred_genre = str(rng.choice(np.array(genres, dtype=object)))
                if preferred_genre is not None:
                    candidate = by_culture_genre.get((culture, preferred_genre), [])
                    if not candidate:
                        candidate = pool
                else:
                    candidate = pool
                n_pick = min(int(tracks_per_user), len(candidate))
                if n_pick <= 0:
                    continue
                idx = rng.choice(len(candidate), size=n_pick, replace=False)
                for j in idx.tolist():
                    tid = str(candidate[int(j)]["track_id"])
                    pair = (uid, tid)
                    if pair in seen_pairs:
                        continue
                    seen_pairs.add(pair)
                    w = float(rng.uniform(float(min_weight), float(max_weight)))
                    writer.writerow(
                        {
                            "user_id": uid,
                            "track_id": tid,
                            "weight": w,
                        }
                    )
                    n_rows += 1

    return {
        "metadata": str(meta_path.resolve()),
        "out": str(out_path.resolve()),
        "n_rows": int(n_rows),
        "n_users": int(n_users),
        "n_cultures": int(len(by_culture)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Synthesize weak interactions from metadata.csv.")
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--users_per_culture", type=int, default=20)
    ap.add_argument("--tracks_per_user", type=int, default=50)
    ap.add_argument("--min_weight", type=float, default=0.5)
    ap.add_argument("--max_weight", type=float, default=2.0)
    ap.add_argument("--genre_column", default="label")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = synthesize_interactions(
        metadata_csv=args.metadata,
        out_csv=args.out,
        users_per_culture=int(args.users_per_culture),
        tracks_per_user=int(args.tracks_per_user),
        min_weight=float(args.min_weight),
        max_weight=float(args.max_weight),
        genre_column=args.genre_column,
        seed=int(args.seed),
    )
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
