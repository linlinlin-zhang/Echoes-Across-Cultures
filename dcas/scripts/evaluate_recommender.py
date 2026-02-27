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
from dcas.recommender import recommend_knn, recommend_ot
from dcas.serialization import load_checkpoint


def _safe_mean(x: list[float]) -> float:
    if not x:
        return float("nan")
    return float(mean(x))


def _ci95_bootstrap(values: list[float], samples: int, seed: int) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    arr = np.array(values, dtype=np.float64)
    if arr.size < 2 or int(samples) <= 0:
        m = float(arr.mean())
        return m, m
    rng = np.random.default_rng(int(seed))
    idx = rng.integers(0, int(arr.size), size=(int(samples), int(arr.size)))
    means = arr[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def evaluate_recommender(
    model_path: str | Path,
    tracks_path: str | Path,
    interactions_path: str | Path,
    out_json: str | Path | None = None,
    method: str = "ot",
    k: int = 20,
    epsilon: float = 0.1,
    iters: int = 200,
    prefer_cuda: bool = False,
    bootstrap_samples: int = 2000,
    bootstrap_seed: int = 42,
) -> dict[str, Any]:
    device = torch.device("cuda" if prefer_cuda and torch.cuda.is_available() else "cpu")
    model, _ = load_checkpoint(str(model_path), map_location=str(device))
    tracks = load_tracks(str(tracks_path))
    interactions = load_interactions(str(interactions_path))

    users = sorted({str(i.user_id) for i in interactions})
    cultures = tracks.cultures()

    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    method = str(method).strip().lower()
    if method not in {"ot", "knn"}:
        raise ValueError("method must be one of: ot, knn")
    optional_metric_keys = [
        "cultural_calibration_kl_legacy",
        "target_culture_prob_mean",
        "user_culture_alignment_kl",
    ]
    for u in users:
        for c in cultures:
            try:
                if method == "ot":
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
                else:
                    _, metrics = recommend_knn(
                        model=model,
                        tracks=tracks,
                        interactions=interactions,
                        user_id=u,
                        target_culture=c,
                        k=int(k),
                        device=device,
                    )
                row = {
                    "user_id": u,
                    "target_culture": c,
                    "serendipity": float(metrics["serendipity"]),
                    "cultural_calibration_kl": float(metrics["cultural_calibration_kl"]),
                }
                for mk in optional_metric_keys:
                    if mk in metrics:
                        row[mk] = float(metrics[mk])
                rows.append(row)
            except Exception as e:
                skipped.append({"user_id": u, "target_culture": c, "reason": str(e)})

    ser = [float(r["serendipity"]) for r in rows]
    ckl = [float(r["cultural_calibration_kl"]) for r in rows]
    legacy = [float(r["cultural_calibration_kl_legacy"]) for r in rows if "cultural_calibration_kl_legacy" in r]
    target_prob = [float(r["target_culture_prob_mean"]) for r in rows if "target_culture_prob_mean" in r]
    user_align_kl = [float(r["user_culture_alignment_kl"]) for r in rows if "user_culture_alignment_kl" in r]

    per_culture: dict[str, dict[str, float]] = {}
    tmp: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"ser": [], "ckl": [], "legacy": [], "target_prob": [], "user_align_kl": []}
    )
    for r in rows:
        c = str(r["target_culture"])
        tmp[c]["ser"].append(float(r["serendipity"]))
        tmp[c]["ckl"].append(float(r["cultural_calibration_kl"]))
        if "cultural_calibration_kl_legacy" in r:
            tmp[c]["legacy"].append(float(r["cultural_calibration_kl_legacy"]))
        if "target_culture_prob_mean" in r:
            tmp[c]["target_prob"].append(float(r["target_culture_prob_mean"]))
        if "user_culture_alignment_kl" in r:
            tmp[c]["user_align_kl"].append(float(r["user_culture_alignment_kl"]))
    for c in sorted(tmp.keys()):
        ser_c = tmp[c]["ser"]
        ckl_c = tmp[c]["ckl"]
        ser_ci_l, ser_ci_h = _ci95_bootstrap(ser_c, samples=int(bootstrap_samples), seed=int(bootstrap_seed) + 13)
        ckl_ci_l, ckl_ci_h = _ci95_bootstrap(ckl_c, samples=int(bootstrap_samples), seed=int(bootstrap_seed) + 29)
        per_culture[c] = {
            "n": int(len(ser_c)),
            "serendipity_mean": _safe_mean(ser_c),
            "serendipity_std": float(np.std(np.array(ser_c, dtype=np.float64))) if ser_c else float("nan"),
            "serendipity_ci95_low": float(ser_ci_l),
            "serendipity_ci95_high": float(ser_ci_h),
            "cultural_calibration_kl_mean": _safe_mean(ckl_c),
            "cultural_calibration_kl_std": float(np.std(np.array(ckl_c, dtype=np.float64))) if ckl_c else float("nan"),
            "cultural_calibration_kl_ci95_low": float(ckl_ci_l),
            "cultural_calibration_kl_ci95_high": float(ckl_ci_h),
        }
        if tmp[c]["legacy"]:
            per_culture[c]["cultural_calibration_kl_legacy_mean"] = _safe_mean(tmp[c]["legacy"])
        if tmp[c]["target_prob"]:
            per_culture[c]["target_culture_prob_mean"] = _safe_mean(tmp[c]["target_prob"])
        if tmp[c]["user_align_kl"]:
            per_culture[c]["user_culture_alignment_kl_mean"] = _safe_mean(tmp[c]["user_align_kl"])

    ser_ci_l, ser_ci_h = _ci95_bootstrap(ser, samples=int(bootstrap_samples), seed=int(bootstrap_seed))
    ckl_ci_l, ckl_ci_h = _ci95_bootstrap(ckl, samples=int(bootstrap_samples), seed=int(bootstrap_seed) + 1)

    result: dict[str, Any] = {
        "summary": {
            "n_users": int(len(users)),
            "n_cultures": int(len(cultures)),
            "n_user_culture_evals": int(len(rows)),
            "n_skipped": int(len(skipped)),
            "serendipity_mean": _safe_mean(ser),
            "serendipity_std": float(np.std(np.array(ser, dtype=np.float64))) if ser else float("nan"),
            "serendipity_ci95_low": float(ser_ci_l),
            "serendipity_ci95_high": float(ser_ci_h),
            "cultural_calibration_kl_mean": _safe_mean(ckl),
            "cultural_calibration_kl_std": float(np.std(np.array(ckl, dtype=np.float64))) if ckl else float("nan"),
            "cultural_calibration_kl_ci95_low": float(ckl_ci_l),
            "cultural_calibration_kl_ci95_high": float(ckl_ci_h),
        },
        "per_target_culture": per_culture,
        "rows": rows,
        "skipped": skipped[:200],
        "config": {
            "method": method,
            "k": int(k),
            "epsilon": float(epsilon),
            "iters": int(iters),
            "device": str(device),
            "bootstrap_samples": int(bootstrap_samples),
            "bootstrap_seed": int(bootstrap_seed),
        },
    }
    if legacy:
        result["summary"]["cultural_calibration_kl_legacy_mean"] = _safe_mean(legacy)
        result["summary"]["cultural_calibration_kl_legacy_std"] = (
            float(np.std(np.array(legacy, dtype=np.float64))) if legacy else float("nan")
        )
    if target_prob:
        result["summary"]["target_culture_prob_mean"] = _safe_mean(target_prob)
        result["summary"]["target_culture_prob_std"] = (
            float(np.std(np.array(target_prob, dtype=np.float64))) if target_prob else float("nan")
        )
    if user_align_kl:
        result["summary"]["user_culture_alignment_kl_mean"] = _safe_mean(user_align_kl)
        result["summary"]["user_culture_alignment_kl_std"] = (
            float(np.std(np.array(user_align_kl, dtype=np.float64))) if user_align_kl else float("nan")
        )

    if out_json is not None:
        out = Path(out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate recommender over all users x target cultures.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--interactions", required=True)
    ap.add_argument("--out_json", default=None)
    ap.add_argument("--method", default="ot", choices=["ot", "knn"])
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--epsilon", type=float, default=0.1)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--bootstrap_samples", type=int, default=2000)
    ap.add_argument("--bootstrap_seed", type=int, default=42)
    ap.add_argument("--prefer_cuda", action="store_true")
    args = ap.parse_args()

    out = evaluate_recommender(
        model_path=args.model,
        tracks_path=args.tracks,
        interactions_path=args.interactions,
        out_json=args.out_json,
        method=str(args.method),
        k=int(args.k),
        epsilon=float(args.epsilon),
        iters=int(args.iters),
        bootstrap_samples=int(args.bootstrap_samples),
        bootstrap_seed=int(args.bootstrap_seed),
        prefer_cuda=bool(args.prefer_cuda),
    )
    print(json.dumps(out["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
