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
    mode: str = "single_culture",
    secondary_cultures: int = 2,
    home_share: float = 0.65,
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
    cultures_sorted = sorted(by_culture.keys())

    def _pick_rows(pool: list[dict[str, str]], n_pick: int, preferred_genre: str | None) -> list[dict[str, str]]:
        if not pool or n_pick <= 0:
            return []
        candidate = pool
        if preferred_genre is not None and has_genre:
            culture_name = str(pool[0].get("culture", "")).strip()
            maybe = by_culture_genre.get((culture_name, preferred_genre), [])
            if maybe:
                candidate = maybe
        n_take = min(int(n_pick), len(candidate))
        if n_take <= 0:
            return []
        idx = rng.choice(len(candidate), size=n_take, replace=False)
        return [candidate[int(j)] for j in idx.tolist()]

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["user_id", "track_id", "weight"])
        writer.writeheader()

        for culture in cultures_sorted:
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
                if str(mode).strip().lower() == "mixed_culture":
                    n_other = min(int(secondary_cultures), max(0, len(cultures_sorted) - 1))
                    others = [c for c in cultures_sorted if c != culture]
                    picked_others = []
                    if n_other > 0 and others:
                        chosen = rng.choice(np.array(others, dtype=object), size=n_other, replace=False)
                        picked_others = [str(x) for x in np.atleast_1d(chosen).tolist()]
                    total_pick = min(int(tracks_per_user), len(rows))
                    home_n = int(round(float(total_pick) * float(np.clip(home_share, 0.0, 1.0))))
                    other_n = max(0, total_pick - home_n)
                    selected_rows = _pick_rows(pool=pool, n_pick=home_n, preferred_genre=preferred_genre)
                    if picked_others and other_n > 0:
                        splits = np.full((len(picked_others),), other_n // len(picked_others), dtype=np.int64)
                        splits[: other_n % len(picked_others)] += 1
                        for other_culture, n_take in zip(picked_others, splits.tolist()):
                            other_pool = by_culture.get(str(other_culture), [])
                            other_genres = sorted(
                                {
                                    str(x.get(genre_column, "")).strip()
                                    for x in other_pool
                                    if str(x.get(genre_column, "")).strip() != ""
                                }
                            )
                            other_pref = None
                            if has_genre and other_genres:
                                other_pref = str(rng.choice(np.array(other_genres, dtype=object)))
                            selected_rows.extend(_pick_rows(pool=other_pool, n_pick=int(n_take), preferred_genre=other_pref))
                else:
                    selected_rows = _pick_rows(pool=pool, n_pick=int(tracks_per_user), preferred_genre=preferred_genre)
                for row in selected_rows:
                    tid = str(row["track_id"])
                    pair = (uid, tid)
                    if pair in seen_pairs:
                        continue
                    seen_pairs.add(pair)
                    w = float(rng.uniform(float(min_weight), float(max_weight)))
                    if str(mode).strip().lower() == "mixed_culture":
                        row_culture = str(row.get("culture", "")).strip()
                        if row_culture == culture:
                            w *= 1.15
                        else:
                            w *= 0.9
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
    ap.add_argument("--mode", default="single_culture", choices=["single_culture", "mixed_culture"])
    ap.add_argument("--secondary_cultures", type=int, default=2)
    ap.add_argument("--home_share", type=float, default=0.65)
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
        mode=str(args.mode),
        secondary_cultures=int(args.secondary_cultures),
        home_share=float(args.home_share),
        seed=int(args.seed),
    )
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
