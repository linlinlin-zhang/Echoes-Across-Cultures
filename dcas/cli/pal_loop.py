from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from dcas.data.npz_tracks import load_tracks
from dcas.pal.uncertainty import rank_by_uncertainty
from dcas.serialization import load_checkpoint


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--prefer_cuda", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if args.prefer_cuda and torch.cuda.is_available() else "cpu")
    model, _ = load_checkpoint(args.model, map_location=str(device))
    tracks = load_tracks(args.tracks)

    ranked = rank_by_uncertainty(model=model, tracks=tracks, device=device)
    top = ranked[: int(args.n)]

    track_id_to_idx = {str(tid): i for i, tid in enumerate(tracks.track_id.tolist())}
    x_all = torch.from_numpy(tracks.embedding.astype(np.float32)).to(device)
    model.eval()
    model.to(device)
    with torch.no_grad():
        _, _, za_mu = model.encode(x_all)

    with open(args.out, "w", encoding="utf-8") as f:
        for tid, score in top:
            idx = track_id_to_idx[tid]
            z = za_mu[idx : idx + 1]
            d = torch.cdist(z, za_mu).squeeze(0)
            nn = int(torch.topk(d, k=6, largest=False).indices[1].item())
            obj = {
                "track_id": tid,
                "culture": str(tracks.culture[idx]),
                "uncertainty": float(score),
                "compare_to": str(tracks.track_id[nn]),
                "question": "它们在情感/功能上是否相似？如果相似/不相似，请给出理由（rationale）。",
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()

