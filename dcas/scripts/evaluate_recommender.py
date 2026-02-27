from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
import torch

from dcas.data.interactions import load_interactions
from dcas.data.npz_tracks import load_tracks
from dcas.recommender import recommend_ot
from dcas.serialization import load_checkpoint


def _safe_mean(x: list[float]) -> float:
    if not x:
        return float("nan")
    return float(mean(x))


def evaluate_recommender(
    model_path: str | Path,
    tracks_path: str | Path,
    interactions_path: str | Path,
    out_json: str | Path | None = None,
    k: int = 20,
    epsilon: float = 0.1,
    iters: int = 200,
    prefer_cuda: bool = False,
) -> dict[str, Any]:
    device = torch.device("cuda" if prefer_cuda and torch.cuda.is_available() else "cpu")
    model, _ = load_checkpoint(str(model_path), map_location=str(device))
    tracks = load_tracks(str(tracks_path))
    interactions = load_interactions(str(interactions_path))

    users = sorted({str(i.user_id) for i in interactions})
    cultures = tracks.cultures()

    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for u in users:
        for c in cultures:
            try:
                _, metrics = recommend_ot(
                    model=model,
                    tracks=tracks,
                    interactions=interactions,
                    user_id=u,
                    target_culture=c,
                    k=int(k),
                    device=device,
                    epsilon=float(epsilon),
                    iters=int(iters),
                )
                rows.append(
                    {
                        "user_id": u,
                        "target_culture": c,
                        "serendipity": float(metrics["serendipity"]),
                        "cultural_calibration_kl": float(metrics["cultural_calibration_kl"]),
                    }
                )
            except Exception as e:
                skipped.append({"user_id": u, "target_culture": c, "reason": str(e)})

    ser = [float(r["serendipity"]) for r in rows]
    ckl = [float(r["cultural_calibration_kl"]) for r in rows]

    per_culture: dict[str, dict[str, float]] = {}
    tmp: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"ser": [], "ckl": []})
    for r in rows:
        c = str(r["target_culture"])
        tmp[c]["ser"].append(float(r["serendipity"]))
        tmp[c]["ckl"].append(float(r["cultural_calibration_kl"]))
    for c in sorted(tmp.keys()):
        per_culture[c] = {
            "n": int(len(tmp[c]["ser"])),
            "serendipity_mean": _safe_mean(tmp[c]["ser"]),
            "cultural_calibration_kl_mean": _safe_mean(tmp[c]["ckl"]),
        }

    result: dict[str, Any] = {
        "summary": {
            "n_users": int(len(users)),
            "n_cultures": int(len(cultures)),
            "n_user_culture_evals": int(len(rows)),
            "n_skipped": int(len(skipped)),
            "serendipity_mean": _safe_mean(ser),
            "serendipity_std": float(np.std(np.array(ser, dtype=np.float64))) if ser else float("nan"),
            "cultural_calibration_kl_mean": _safe_mean(ckl),
            "cultural_calibration_kl_std": float(np.std(np.array(ckl, dtype=np.float64))) if ckl else float("nan"),
        },
        "per_target_culture": per_culture,
        "rows": rows,
        "skipped": skipped[:200],
        "config": {
            "k": int(k),
            "epsilon": float(epsilon),
            "iters": int(iters),
            "device": str(device),
        },
    }

    if out_json is not None:
        out = Path(out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate OT recommender over all users x target cultures.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--interactions", required=True)
    ap.add_argument("--out_json", default=None)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--epsilon", type=float, default=0.1)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--prefer_cuda", action="store_true")
    args = ap.parse_args()

    out = evaluate_recommender(
        model_path=args.model,
        tracks_path=args.tracks,
        interactions_path=args.interactions,
        out_json=args.out_json,
        k=int(args.k),
        epsilon=float(args.epsilon),
        iters=int(args.iters),
        prefer_cuda=bool(args.prefer_cuda),
    )
    print(json.dumps(out["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
