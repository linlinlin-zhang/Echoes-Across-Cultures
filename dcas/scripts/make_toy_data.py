from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def generate_toy_data(out_dir: str | Path, n_tracks: int = 3000, dim: int = 128, seed: int = 7) -> Path:
    rng = np.random.default_rng(int(seed))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cultures = ["west", "india", "africa"]
    n_affect = 6
    zc_dim, zs_dim, za_dim = 16, 16, 8

    Wc = rng.normal(scale=0.6, size=(zc_dim, int(dim))).astype(np.float32)
    Ws = rng.normal(scale=0.6, size=(zs_dim, int(dim))).astype(np.float32)
    Wa = rng.normal(scale=0.6, size=(za_dim, int(dim))).astype(np.float32)

    style_centers = rng.normal(size=(len(cultures), zs_dim)).astype(np.float32)
    affect_centers = rng.normal(size=(n_affect, za_dim)).astype(np.float32)

    N = int(n_tracks)
    track_id = np.array([f"t{i:05d}" for i in range(N)], dtype="<U16")
    culture = rng.choice(np.array(cultures, dtype="<U16"), size=(N,), replace=True)
    affect_label = rng.integers(0, n_affect, size=(N,), dtype=np.int64)

    zc = rng.normal(size=(N, zc_dim)).astype(np.float32)
    zs = rng.normal(scale=0.3, size=(N, zs_dim)).astype(np.float32)
    za = rng.normal(scale=0.3, size=(N, za_dim)).astype(np.float32)

    culture_to_id = {c: i for i, c in enumerate(cultures)}
    for i in range(N):
        zs[i] += style_centers[culture_to_id[str(culture[i])]]
        za[i] += affect_centers[int(affect_label[i])]

    emb = (zc @ Wc + zs @ Ws + za @ Wa).astype(np.float32)
    emb += rng.normal(scale=0.2, size=emb.shape).astype(np.float32)

    np.savez_compressed(
        str(out_dir / "tracks.npz"),
        track_id=track_id,
        culture=culture,
        embedding=emb.astype(np.float32),
        affect_label=affect_label,
    )

    users = [f"u{i}" for i in range(6)]
    user_pref = {
        "u0": ("west", [0, 1]),
        "u1": ("west", [2, 3]),
        "u2": ("india", [0, 4]),
        "u3": ("india", [1, 5]),
        "u4": ("africa", [2, 4]),
        "u5": ("africa", [3, 5]),
    }

    with open(out_dir / "interactions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["user_id", "track_id", "weight"])
        w.writeheader()
        for u in users:
            cul, affects = user_pref[u]
            idx = np.nonzero((culture == cul) & np.isin(affect_label, np.array(affects, dtype=np.int64)))[0]
            pick = rng.choice(idx, size=80, replace=False)
            for j in pick.tolist():
                w.writerow({"user_id": u, "track_id": str(track_id[j]), "weight": float(rng.uniform(0.5, 2.0))})

    with open(out_dir / "meta.txt", "w", encoding="utf-8") as f:
        f.write(f"cultures={cultures}\n")
        f.write(f"n_affect={n_affect}\n")
        f.write(f"dim={int(dim)}\n")

    return out_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_tracks", type=int, default=3000)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    out_dir = generate_toy_data(out_dir=args.out, n_tracks=int(args.n_tracks), dim=int(args.dim), seed=int(args.seed))
    print(str(out_dir))


if __name__ == "__main__":
    main()

