from __future__ import annotations

import argparse
import json

from dcas.pipelines import style_transfer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="model checkpoint")
    ap.add_argument("--tracks", required=True, help="tracks.npz")
    ap.add_argument("--source_track", required=True, help="source track id")
    ap.add_argument("--style_track", required=True, help="style donor track id")
    ap.add_argument("--out", required=True, help="output npz artifact")
    ap.add_argument("--target_culture", default=None, help="optional candidate filter culture")
    ap.add_argument("--alpha", type=float, default=1.0, help="style mixing ratio")
    ap.add_argument("--k", type=int, default=10, help="number of nearest neighbors to return")
    ap.add_argument("--prefer_cuda", action="store_true")
    args = ap.parse_args()

    out = style_transfer(
        model_path=args.model,
        tracks_path=args.tracks,
        source_track_id=args.source_track,
        style_track_id=args.style_track,
        out_path=args.out,
        target_culture=args.target_culture,
        alpha=float(args.alpha),
        k=int(args.k),
        prefer_cuda=bool(args.prefer_cuda),
    )
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()

