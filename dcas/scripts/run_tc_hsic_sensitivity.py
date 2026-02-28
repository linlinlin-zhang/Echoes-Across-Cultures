from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from dcas.pipelines import train_model
from dcas.scripts.evaluate_recommender import evaluate_recommender


def _parse_float_csv(raw: str, name: str) -> list[float]:
    out = [float(x.strip()) for x in str(raw).split(",") if x.strip()]
    if not out:
        raise ValueError(f"{name} cannot be empty")
    return out


def _parse_int_csv(raw: str, name: str) -> list[int]:
    out = [int(x.strip()) for x in str(raw).split(",") if x.strip()]
    if not out:
        raise ValueError(f"{name} cannot be empty")
    return out


def _slug_float(v: float) -> str:
    return f"{float(v):.6g}".replace("-", "m").replace(".", "p")


def _stats(vals: list[float]) -> dict[str, float]:
    arr = np.array(vals, dtype=np.float64)
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "ci95": float("nan")}
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    ci95 = float(1.96 * std / math.sqrt(float(arr.size))) if arr.size > 1 else 0.0
    return {"mean": mean, "std": std, "ci95": ci95}


def _safe_z(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float64)
    mu = float(arr.mean())
    sd = float(arr.std(ddof=0))
    if sd <= 1e-12:
        return np.zeros_like(arr)
    return (arr - mu) / sd


def _rank_desc(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(-scores)
    rank = np.empty_like(order, dtype=np.float64)
    rank[order] = np.arange(int(scores.size), dtype=np.float64)
    return rank


def _spearman_from_scores(a: np.ndarray, b: np.ndarray) -> float:
    if a.size != b.size or a.size < 2:
        return float("nan")
    ra = _rank_desc(a)
    rb = _rank_desc(b)
    sa = float(np.std(ra))
    sb = float(np.std(rb))
    if sa <= 1e-12 or sb <= 1e-12:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def run_tc_hsic_sensitivity(
    tracks_path: str,
    interactions_path: str,
    out_dir: str,
    model_dir: str,
    tc_values: list[float],
    hsic_values: list[float],
    seeds: list[int],
    constraints_path: str | None = None,
    epochs: int = 10,
    batch_size: int = 64,
    lr: float = 2e-3,
    lambda_constraints: float = 0.2,
    constraint_margin: float = 1.0,
    lambda_domain: float = 0.5,
    lambda_contrast: float = 0.2,
    lambda_cov: float = 0.05,
    beta_kl: float = 1.0,
    shared_encoder: bool = False,
    regularizer_warmup_epochs: int = 0,
    k: int = 10,
    epsilon: float = 0.1,
    iters: int = 200,
    bootstrap_samples: int = 2000,
    minority_quantile: float = 0.25,
    prefer_cuda: bool = False,
) -> dict[str, Any]:
    out_dir_p = Path(out_dir)
    model_dir_p = Path(model_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)
    model_dir_p.mkdir(parents=True, exist_ok=True)

    runs: list[dict[str, Any]] = []
    by_cell: dict[tuple[float, float], list[dict[str, Any]]] = {}
    cell_order = [(float(tc), float(hs)) for tc in tc_values for hs in hsic_values]

    for tc in tc_values:
        for hsic in hsic_values:
            key = (float(tc), float(hsic))
            by_cell[key] = []
            for seed in seeds:
                tc_s = _slug_float(float(tc))
                hs_s = _slug_float(float(hsic))
                model_path = model_dir_p / f"tc_{tc_s}__hsic_{hs_s}__seed_{int(seed)}.pt"
                eval_path = out_dir_p / f"eval_tc_{tc_s}__hsic_{hs_s}__seed_{int(seed)}.json"

                train_model(
                    tracks_path=tracks_path,
                    out_path=model_path,
                    constraints_path=constraints_path,
                    epochs=int(epochs),
                    batch_size=int(batch_size),
                    lr=float(lr),
                    seed=int(seed),
                    prefer_cuda=bool(prefer_cuda),
                    lambda_constraints=float(lambda_constraints),
                    constraint_margin=float(constraint_margin),
                    lambda_domain=float(lambda_domain),
                    lambda_contrast=float(lambda_contrast),
                    lambda_cov=float(lambda_cov),
                    lambda_tc=float(tc),
                    lambda_hsic=float(hsic),
                    beta_kl=float(beta_kl),
                    shared_encoder=bool(shared_encoder),
                    regularizer_warmup_epochs=int(regularizer_warmup_epochs),
                )

                ev = evaluate_recommender(
                    model_path=str(model_path),
                    tracks_path=tracks_path,
                    interactions_path=interactions_path,
                    out_json=str(eval_path),
                    method="ot",
                    k=int(k),
                    epsilon=float(epsilon),
                    iters=int(iters),
                    bootstrap_samples=int(bootstrap_samples),
                    bootstrap_seed=int(seed) + 97,
                    minority_quantile=float(minority_quantile),
                    prefer_cuda=bool(prefer_cuda),
                )

                row = {
                    "lambda_tc": float(tc),
                    "lambda_hsic": float(hsic),
                    "seed": int(seed),
                    "model_path": str(model_path),
                    "eval_path": str(eval_path),
                    "serendipity_mean": float(ev["summary"]["serendipity_mean"]),
                    "cultural_calibration_kl_mean": float(ev["summary"]["cultural_calibration_kl_mean"]),
                    "minority_exposure_at_k_mean": float(ev["summary"].get("minority_exposure_at_k_mean", float("nan"))),
                }
                runs.append(row)
                by_cell[key].append(row)

    grid_rows: list[dict[str, Any]] = []
    for tc, hsic in cell_order:
        cell = by_cell[(tc, hsic)]
        ser = [float(r["serendipity_mean"]) for r in cell]
        ckl = [float(r["cultural_calibration_kl_mean"]) for r in cell]
        minority = [float(r["minority_exposure_at_k_mean"]) for r in cell]
        grid_rows.append(
            {
                "lambda_tc": float(tc),
                "lambda_hsic": float(hsic),
                "n_runs": int(len(cell)),
                "serendipity": _stats(ser),
                "cultural_calibration_kl": _stats(ckl),
                "minority_exposure_at_k": _stats(minority),
            }
        )

    ser_means = np.array([float(r["serendipity"]["mean"]) for r in grid_rows], dtype=np.float64)
    ckl_means = np.array([float(r["cultural_calibration_kl"]["mean"]) for r in grid_rows], dtype=np.float64)
    min_means = np.array([float(r["minority_exposure_at_k"]["mean"]) for r in grid_rows], dtype=np.float64)
    objective = _safe_z(ser_means) - _safe_z(ckl_means) + _safe_z(min_means)
    for i, r in enumerate(grid_rows):
        r["objective_zscore"] = float(objective[i])

    best_idx = int(np.argmax(objective))
    global_best = {
        "lambda_tc": float(grid_rows[best_idx]["lambda_tc"]),
        "lambda_hsic": float(grid_rows[best_idx]["lambda_hsic"]),
        "objective_zscore": float(objective[best_idx]),
    }

    seed_scores: dict[int, np.ndarray] = {}
    best_by_seed: dict[str, str] = {}
    best_counts: dict[str, int] = {}
    for seed in seeds:
        rows_seed = [r for r in runs if int(r["seed"]) == int(seed)]
        rows_seed_map = {(float(r["lambda_tc"]), float(r["lambda_hsic"])): r for r in rows_seed}
        if any(k not in rows_seed_map for k in cell_order):
            continue
        ser = np.array([float(rows_seed_map[k]["serendipity_mean"]) for k in cell_order], dtype=np.float64)
        ckl = np.array([float(rows_seed_map[k]["cultural_calibration_kl_mean"]) for k in cell_order], dtype=np.float64)
        minority = np.array([float(rows_seed_map[k]["minority_exposure_at_k_mean"]) for k in cell_order], dtype=np.float64)
        score = _safe_z(ser) - _safe_z(ckl) + _safe_z(minority)
        seed_scores[int(seed)] = score
        idx = int(np.argmax(score))
        key = f"tc={cell_order[idx][0]:.6g},hsic={cell_order[idx][1]:.6g}"
        best_by_seed[str(seed)] = key
        best_counts[key] = int(best_counts.get(key, 0) + 1)

    spearman_vals: list[float] = []
    seeds_ok = sorted(seed_scores.keys())
    for i in range(len(seeds_ok)):
        for j in range(i + 1, len(seeds_ok)):
            s1 = seeds_ok[i]
            s2 = seeds_ok[j]
            rho = _spearman_from_scores(seed_scores[s1], seed_scores[s2])
            if not np.isnan(rho):
                spearman_vals.append(float(rho))

    dominant_key = ""
    dominant_freq = 0.0
    if best_counts:
        dominant_key = sorted(best_counts.items(), key=lambda x: (-x[1], x[0]))[0][0]
        dominant_freq = float(max(best_counts.values()) / max(1, len(seed_scores)))

    stability = {
        "seed_count_used": int(len(seed_scores)),
        "best_config_by_seed": best_by_seed,
        "best_config_frequency": best_counts,
        "dominant_best_config": dominant_key,
        "dominant_best_config_ratio": float(dominant_freq),
        "rank_spearman_mean": float(np.mean(np.array(spearman_vals, dtype=np.float64))) if spearman_vals else float("nan"),
        "rank_spearman_std": float(np.std(np.array(spearman_vals, dtype=np.float64))) if spearman_vals else float("nan"),
        "rank_spearman_pairs": int(len(spearman_vals)),
    }

    result: dict[str, Any] = {
        "config": {
            "tracks_path": str(tracks_path),
            "interactions_path": str(interactions_path),
            "constraints_path": str(constraints_path) if constraints_path else None,
            "tc_values": [float(x) for x in tc_values],
            "hsic_values": [float(x) for x in hsic_values],
            "seeds": [int(x) for x in seeds],
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "lr": float(lr),
            "lambda_constraints": float(lambda_constraints),
            "constraint_margin": float(constraint_margin),
            "lambda_domain": float(lambda_domain),
            "lambda_contrast": float(lambda_contrast),
            "lambda_cov": float(lambda_cov),
            "beta_kl": float(beta_kl),
            "shared_encoder": bool(shared_encoder),
            "regularizer_warmup_epochs": int(regularizer_warmup_epochs),
            "k": int(k),
            "epsilon": float(epsilon),
            "iters": int(iters),
            "bootstrap_samples": int(bootstrap_samples),
            "minority_quantile": float(minority_quantile),
            "prefer_cuda": bool(prefer_cuda),
        },
        "global_best_by_objective": global_best,
        "stability": stability,
        "grid": grid_rows,
        "runs": runs,
    }

    out_json = out_dir_p / "tc_hsic_sensitivity_summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    lines = [
        "# TC/HSIC Sensitivity Summary",
        "",
        "| lambda_tc | lambda_hsic | n_runs | serendipity_mean+/-std | calibration_kl_mean+/-std | minority@k_mean+/-std | objective_z |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in sorted(grid_rows, key=lambda x: float(x["objective_zscore"]), reverse=True):
        ser = r["serendipity"]
        ckl = r["cultural_calibration_kl"]
        minority = r["minority_exposure_at_k"]
        lines.append(
            f"| {float(r['lambda_tc']):.6g} | {float(r['lambda_hsic']):.6g} | {int(r['n_runs'])} | "
            f"{float(ser['mean']):.6f} +/- {float(ser['std']):.6f} | "
            f"{float(ckl['mean']):.6f} +/- {float(ckl['std']):.6f} | "
            f"{float(minority['mean']):.6f} +/- {float(minority['std']):.6f} | "
            f"{float(r['objective_zscore']):+.4f} |"
        )
    lines.extend(
        [
            "",
            "## Stability Evidence",
            "",
            f"- seed_count_used: `{int(stability['seed_count_used'])}`",
            f"- dominant_best_config: `{stability['dominant_best_config']}`",
            f"- dominant_best_config_ratio: `{float(stability['dominant_best_config_ratio']):.6f}`",
            f"- rank_spearman_mean: `{float(stability['rank_spearman_mean']):.6f}`",
            f"- rank_spearman_std: `{float(stability['rank_spearman_std']):.6f}`",
            f"- rank_spearman_pairs: `{int(stability['rank_spearman_pairs'])}`",
            "",
        ]
    )
    out_md = out_dir_p / "tc_hsic_sensitivity_summary.md"
    out_md.write_text("\n".join(lines), encoding="utf-8")

    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Grid-search lambda_tc x lambda_hsic with multi-seed stability report.")
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--interactions", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--constraints", default=None)
    ap.add_argument("--tc_values", default="0.0,0.02,0.05,0.1")
    ap.add_argument("--hsic_values", default="0.0,0.01,0.02,0.05")
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--lambda_constraints", type=float, default=0.2)
    ap.add_argument("--constraint_margin", type=float, default=1.0)
    ap.add_argument("--lambda_domain", type=float, default=0.5)
    ap.add_argument("--lambda_contrast", type=float, default=0.2)
    ap.add_argument("--lambda_cov", type=float, default=0.05)
    ap.add_argument("--beta_kl", type=float, default=1.0)
    ap.add_argument("--shared_encoder", action="store_true")
    ap.add_argument("--regularizer_warmup_epochs", type=int, default=0)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--epsilon", type=float, default=0.1)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--bootstrap_samples", type=int, default=2000)
    ap.add_argument("--minority_quantile", type=float, default=0.25)
    ap.add_argument("--prefer_cuda", action="store_true")
    args = ap.parse_args()

    out = run_tc_hsic_sensitivity(
        tracks_path=str(args.tracks),
        interactions_path=str(args.interactions),
        out_dir=str(args.out_dir),
        model_dir=str(args.model_dir),
        constraints_path=str(args.constraints) if args.constraints else None,
        tc_values=_parse_float_csv(str(args.tc_values), name="tc_values"),
        hsic_values=_parse_float_csv(str(args.hsic_values), name="hsic_values"),
        seeds=_parse_int_csv(str(args.seeds), name="seeds"),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        lambda_constraints=float(args.lambda_constraints),
        constraint_margin=float(args.constraint_margin),
        lambda_domain=float(args.lambda_domain),
        lambda_contrast=float(args.lambda_contrast),
        lambda_cov=float(args.lambda_cov),
        beta_kl=float(args.beta_kl),
        shared_encoder=bool(args.shared_encoder),
        regularizer_warmup_epochs=int(args.regularizer_warmup_epochs),
        k=int(args.k),
        epsilon=float(args.epsilon),
        iters=int(args.iters),
        bootstrap_samples=int(args.bootstrap_samples),
        minority_quantile=float(args.minority_quantile),
        prefer_cuda=bool(args.prefer_cuda),
    )
    tiny = {
        "global_best_by_objective": out["global_best_by_objective"],
        "stability": out["stability"],
    }
    print(json.dumps(tiny, ensure_ascii=False))


if __name__ == "__main__":
    main()
