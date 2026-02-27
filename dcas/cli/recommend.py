from __future__ import annotations

import argparse
import json

import torch

from dcas.data.interactions import load_interactions
from dcas.data.npz_tracks import load_tracks
from dcas.recommender import recommend_knn, recommend_ot
from dcas.serialization import load_checkpoint


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--interactions", required=True)
    ap.add_argument("--user", required=True)
    ap.add_argument("--target_culture", required=True)
    ap.add_argument("--method", default="ot", choices=["ot", "knn"])
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--prefer_cuda", action="store_true")
    ap.add_argument("--epsilon", type=float, default=0.1)
    ap.add_argument("--iters", type=int, default=200)
    args = ap.parse_args()

    device = torch.device("cuda" if args.prefer_cuda and torch.cuda.is_available() else "cpu")
    model, _ = load_checkpoint(args.model, map_location=str(device))

    tracks = load_tracks(args.tracks)
    interactions = load_interactions(args.interactions)

    if args.method == "ot":
        recs, metrics = recommend_ot(
            model=model,
            tracks=tracks,
            interactions=interactions,
            user_id=args.user,
            target_culture=args.target_culture,
            k=int(args.k),
            device=device,
            epsilon=float(args.epsilon),
            iters=int(args.iters),
        )
    else:
        recs, metrics = recommend_knn(
            model=model,
            tracks=tracks,
            interactions=interactions,
            user_id=args.user,
            target_culture=args.target_culture,
            k=int(args.k),
            device=device,
        )

    print(json.dumps(metrics, ensure_ascii=False))
    for r in recs:
        print(
            json.dumps(
                {
                    "track_id": r.track_id,
                    "culture": r.culture,
                    "score": r.score,
                    "relevance": r.relevance,
                    "unexpectedness": r.unexpectedness,
                },
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()

