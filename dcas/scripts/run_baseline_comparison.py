from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from dcas.pipelines import train_model
from dcas.scripts.compare_recommender_runs import compare_recommender_runs
from dcas.scripts.evaluate_recommender import evaluate_recommender


def _parse_int_csv(raw: str, name: str) -> list[int]:
    out = [int(x.strip()) for x in str(raw).split(",") if x.strip()]
    if not out:
        raise ValueError(f"{name} cannot be empty")
    return out


def _stats(vals: list[float]) -> dict[str, float]:
    arr = np.array(vals, dtype=np.float64)
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "ci95": float("nan")}
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    ci95 = float(1.96 * std / math.sqrt(float(arr.size))) if arr.size > 1 else 0.0
    return {"mean": mean, "std": std, "ci95": ci95}


def run_baseline_comparison(
    tracks_path: str,
    interactions_path: str,
    out_dir: str,
    model_dir: str,
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
    lambda_tc: float = 0.05,
    lambda_hsic: float = 0.02,
    beta_kl: float = 1.0,
    beta_vae_beta: float = 4.0,
    factorvae_lambda_tc: float = 0.1,
    regularizer_warmup_epochs: int = 0,
    k: int = 10,
    epsilon: float = 0.1,
    iters: int = 200,
    bootstrap_samples: int = 2000,
    permutation_samples: int = 2000,
    minority_quantile: float = 0.25,
    prefer_cuda: bool = False,
) -> dict[str, Any]:
    out_dir_p = Path(out_dir)
    model_dir_p = Path(model_dir)
    compare_dir_p = out_dir_p / "comparisons"
    out_dir_p.mkdir(parents=True, exist_ok=True)
    model_dir_p.mkdir(parents=True, exist_ok=True)
    compare_dir_p.mkdir(parents=True, exist_ok=True)

    full_name = "three_factor_dcas"
    variants: list[dict[str, Any]] = [
        {
            "name": full_name,
            "shared_encoder": False,
            "beta_kl": float(beta_kl),
            "lambda_domain": float(lambda_domain),
            "lambda_contrast": float(lambda_contrast),
            "lambda_cov": float(lambda_cov),
            "lambda_tc": float(lambda_tc),
            "lambda_hsic": float(lambda_hsic),
            "constraints_path": str(constraints_path) if constraints_path else None,
            "lambda_constraints": float(lambda_constraints) if constraints_path else 0.0,
        },
        {
            "name": "vae",
            "shared_encoder": True,
            "beta_kl": 1.0,
            "lambda_domain": 0.0,
            "lambda_contrast": 0.0,
            "lambda_cov": 0.0,
            "lambda_tc": 0.0,
            "lambda_hsic": 0.0,
            "constraints_path": None,
            "lambda_constraints": 0.0,
        },
        {
            "name": "beta_vae",
            "shared_encoder": True,
            "beta_kl": float(beta_vae_beta),
            "lambda_domain": 0.0,
            "lambda_contrast": 0.0,
            "lambda_cov": 0.0,
            "lambda_tc": 0.0,
            "lambda_hsic": 0.0,
            "constraints_path": None,
            "lambda_constraints": 0.0,
        },
        {
            "name": "factorvae",
            "shared_encoder": True,
            "beta_kl": 1.0,
            "lambda_domain": 0.0,
            "lambda_contrast": 0.0,
            "lambda_cov": 0.0,
            "lambda_tc": float(factorvae_lambda_tc),
            "lambda_hsic": 0.0,
            "constraints_path": None,
            "lambda_constraints": 0.0,
        },
    ]

    runs: list[dict[str, Any]] = []
    eval_path_map: dict[tuple[str, int], str] = {}
    for variant in variants:
        name = str(variant["name"])
        for seed in seeds:
            model_path = model_dir_p / f"{name}__seed_{int(seed)}.pt"
            eval_path = out_dir_p / f"eval_{name}__seed_{int(seed)}.json"
            train_model(
                tracks_path=tracks_path,
                out_path=str(model_path),
                constraints_path=str(variant["constraints_path"]) if variant["constraints_path"] else None,
                epochs=int(epochs),
                batch_size=int(batch_size),
                lr=float(lr),
                seed=int(seed),
                prefer_cuda=bool(prefer_cuda),
                lambda_constraints=float(variant["lambda_constraints"]),
                constraint_margin=float(constraint_margin),
                lambda_domain=float(variant["lambda_domain"]),
                lambda_contrast=float(variant["lambda_contrast"]),
                lambda_cov=float(variant["lambda_cov"]),
                lambda_tc=float(variant["lambda_tc"]),
                lambda_hsic=float(variant["lambda_hsic"]),
                beta_kl=float(variant["beta_kl"]),
                shared_encoder=bool(variant["shared_encoder"]),
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
                bootstrap_seed=int(seed) + 101,
                minority_quantile=float(minority_quantile),
                prefer_cuda=bool(prefer_cuda),
            )
            row = {
                "variant": name,
                "seed": int(seed),
                "model_path": str(model_path),
                "eval_path": str(eval_path),
                "serendipity_mean": float(ev["summary"]["serendipity_mean"]),
                "cultural_calibration_kl_mean": float(ev["summary"]["cultural_calibration_kl_mean"]),
                "minority_exposure_at_k_mean": float(ev["summary"].get("minority_exposure_at_k_mean", float("nan"))),
            }
            runs.append(row)
            eval_path_map[(name, int(seed))] = str(eval_path)

    by_variant: dict[str, list[dict[str, Any]]] = {}
    for r in runs:
        by_variant.setdefault(str(r["variant"]), []).append(r)

    variant_summary: dict[str, Any] = {}
    for name, rows in by_variant.items():
        ser = [float(r["serendipity_mean"]) for r in rows]
        ckl = [float(r["cultural_calibration_kl_mean"]) for r in rows]
        minority = [float(r["minority_exposure_at_k_mean"]) for r in rows]
        variant_summary[name] = {
            "n_runs": int(len(rows)),
            "serendipity": _stats(ser),
            "cultural_calibration_kl": _stats(ckl),
            "minority_exposure_at_k": _stats(minority),
        }

    metrics = ["serendipity", "cultural_calibration_kl", "minority_exposure_at_k"]
    direction = {"serendipity": 1.0, "cultural_calibration_kl": -1.0, "minority_exposure_at_k": 1.0}
    comparison_vs_full: dict[str, Any] = {}
    for variant in variants:
        name = str(variant["name"])
        if name == full_name:
            continue
        deltas: dict[str, list[float]] = {m: [] for m in metrics}
        pvals: dict[str, list[float]] = {m: [] for m in metrics}
        better: dict[str, list[int]] = {m: [] for m in metrics}
        per_seed: list[dict[str, Any]] = []
        for seed in seeds:
            base_eval = eval_path_map.get((name, int(seed)))
            full_eval = eval_path_map.get((full_name, int(seed)))
            if not base_eval or not full_eval:
                continue
            cmp_json = compare_dir_p / f"compare_{name}_vs_full__seed_{int(seed)}.json"
            cmp_md = compare_dir_p / f"compare_{name}_vs_full__seed_{int(seed)}.md"
            cmp = compare_recommender_runs(
                base_eval_path=base_eval,
                candidate_eval_path=full_eval,
                metrics=metrics,
                bootstrap_samples=int(bootstrap_samples),
                permutation_samples=int(permutation_samples),
                seed=int(seed) + 503,
                out_json=str(cmp_json),
                out_md=str(cmp_md),
            )
            seed_row: dict[str, Any] = {"seed": int(seed)}
            for m in metrics:
                if m not in cmp["metrics"]:
                    continue
                d = float(cmp["metrics"][m]["delta_mean"])
                p = float(cmp["metrics"][m]["p_value_two_sided"])
                deltas[m].append(d)
                pvals[m].append(p)
                better[m].append(1 if direction[m] * d > 0 else 0)
                seed_row[m] = {"delta_mean": d, "p_value_two_sided": p}
            per_seed.append(seed_row)

        metric_summary: dict[str, Any] = {}
        for m in metrics:
            metric_summary[m] = {
                "delta_full_minus_baseline": _stats(deltas[m]),
                "p_value": _stats(pvals[m]),
                "full_better_rate": float(np.mean(np.array(better[m], dtype=np.float64))) if better[m] else float("nan"),
                "n_seeds_compared": int(len(deltas[m])),
            }
        comparison_vs_full[name] = {"metrics": metric_summary, "per_seed": per_seed}

    necessity_checks: dict[str, bool] = {}
    for name in sorted(comparison_vs_full.keys()):
        m = comparison_vs_full[name]["metrics"]
        ser_ok = float(m["serendipity"]["delta_full_minus_baseline"]["mean"]) > 0
        ckl_ok = float(m["cultural_calibration_kl"]["delta_full_minus_baseline"]["mean"]) < 0
        min_ok = float(m["minority_exposure_at_k"]["delta_full_minus_baseline"]["mean"]) > 0
        necessity_checks[name] = bool(ser_ok and ckl_ok and min_ok)
    necessity_checks["all_baselines_support_three_factor"] = bool(all(necessity_checks.values())) if necessity_checks else False

    result: dict[str, Any] = {
        "config": {
            "tracks_path": str(tracks_path),
            "interactions_path": str(interactions_path),
            "constraints_path": str(constraints_path) if constraints_path else None,
            "seeds": [int(s) for s in seeds],
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "lr": float(lr),
            "lambda_constraints": float(lambda_constraints),
            "constraint_margin": float(constraint_margin),
            "lambda_domain": float(lambda_domain),
            "lambda_contrast": float(lambda_contrast),
            "lambda_cov": float(lambda_cov),
            "lambda_tc": float(lambda_tc),
            "lambda_hsic": float(lambda_hsic),
            "beta_kl": float(beta_kl),
            "beta_vae_beta": float(beta_vae_beta),
            "factorvae_lambda_tc": float(factorvae_lambda_tc),
            "regularizer_warmup_epochs": int(regularizer_warmup_epochs),
            "k": int(k),
            "epsilon": float(epsilon),
            "iters": int(iters),
            "bootstrap_samples": int(bootstrap_samples),
            "permutation_samples": int(permutation_samples),
            "minority_quantile": float(minority_quantile),
            "prefer_cuda": bool(prefer_cuda),
        },
        "variants": variants,
        "variant_summary": variant_summary,
        "comparison_vs_three_factor": comparison_vs_full,
        "three_factor_necessity_checks": necessity_checks,
        "runs": runs,
    }

    out_json = out_dir_p / "baseline_comparison_summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    lines = [
        "# Baseline Comparison Draft Table",
        "",
        "| variant | serendipity_mean+/-std | calibration_kl_mean+/-std | minority@k_mean+/-std | delta_ser (full-baseline) | delta_ckl (full-baseline) | delta_minority (full-baseline) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    order = [full_name, "vae", "beta_vae", "factorvae"]
    for name in order:
        s = variant_summary.get(name, {})
        ser = s.get("serendipity", {})
        ckl = s.get("cultural_calibration_kl", {})
        minority = s.get("minority_exposure_at_k", {})
        if name == full_name:
            lines.append(
                f"| {name} | {float(ser.get('mean', float('nan'))):.6f} +/- {float(ser.get('std', float('nan'))):.6f} | "
                f"{float(ckl.get('mean', float('nan'))):.6f} +/- {float(ckl.get('std', float('nan'))):.6f} | "
                f"{float(minority.get('mean', float('nan'))):.6f} +/- {float(minority.get('std', float('nan'))):.6f} | - | - | - |"
            )
            continue
        c = comparison_vs_full.get(name, {}).get("metrics", {})
        d_ser = float(c.get("serendipity", {}).get("delta_full_minus_baseline", {}).get("mean", float("nan")))
        d_ckl = float(c.get("cultural_calibration_kl", {}).get("delta_full_minus_baseline", {}).get("mean", float("nan")))
        d_min = float(c.get("minority_exposure_at_k", {}).get("delta_full_minus_baseline", {}).get("mean", float("nan")))
        lines.append(
            f"| {name} | {float(ser.get('mean', float('nan'))):.6f} +/- {float(ser.get('std', float('nan'))):.6f} | "
            f"{float(ckl.get('mean', float('nan'))):.6f} +/- {float(ckl.get('std', float('nan'))):.6f} | "
            f"{float(minority.get('mean', float('nan'))):.6f} +/- {float(minority.get('std', float('nan'))):.6f} | "
            f"{d_ser:+.6f} | {d_ckl:+.6f} | {d_min:+.6f} |"
        )
    lines.extend(
        [
            "",
            "## Necessity Checks",
            "",
            f"- all_baselines_support_three_factor: `{bool(necessity_checks['all_baselines_support_three_factor'])}`",
            f"- vae: `{bool(necessity_checks.get('vae', False))}`",
            f"- beta_vae: `{bool(necessity_checks.get('beta_vae', False))}`",
            f"- factorvae: `{bool(necessity_checks.get('factorvae', False))}`",
            "",
        ]
    )
    out_md = out_dir_p / "baseline_comparison_table_draft.md"
    out_md.write_text("\n".join(lines), encoding="utf-8")

    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare 3-factor model against VAE/beta-VAE/FactorVAE baselines.")
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--interactions", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--constraints", default=None)
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--lambda_constraints", type=float, default=0.2)
    ap.add_argument("--constraint_margin", type=float, default=1.0)
    ap.add_argument("--lambda_domain", type=float, default=0.5)
    ap.add_argument("--lambda_contrast", type=float, default=0.2)
    ap.add_argument("--lambda_cov", type=float, default=0.05)
    ap.add_argument("--lambda_tc", type=float, default=0.05)
    ap.add_argument("--lambda_hsic", type=float, default=0.02)
    ap.add_argument("--beta_kl", type=float, default=1.0)
    ap.add_argument("--beta_vae_beta", type=float, default=4.0)
    ap.add_argument("--factorvae_lambda_tc", type=float, default=0.1)
    ap.add_argument("--regularizer_warmup_epochs", type=int, default=0)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--epsilon", type=float, default=0.1)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--bootstrap_samples", type=int, default=2000)
    ap.add_argument("--permutation_samples", type=int, default=2000)
    ap.add_argument("--minority_quantile", type=float, default=0.25)
    ap.add_argument("--prefer_cuda", action="store_true")
    args = ap.parse_args()

    out = run_baseline_comparison(
        tracks_path=str(args.tracks),
        interactions_path=str(args.interactions),
        out_dir=str(args.out_dir),
        model_dir=str(args.model_dir),
        constraints_path=str(args.constraints) if args.constraints else None,
        seeds=_parse_int_csv(str(args.seeds), name="seeds"),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        lambda_constraints=float(args.lambda_constraints),
        constraint_margin=float(args.constraint_margin),
        lambda_domain=float(args.lambda_domain),
        lambda_contrast=float(args.lambda_contrast),
        lambda_cov=float(args.lambda_cov),
        lambda_tc=float(args.lambda_tc),
        lambda_hsic=float(args.lambda_hsic),
        beta_kl=float(args.beta_kl),
        beta_vae_beta=float(args.beta_vae_beta),
        factorvae_lambda_tc=float(args.factorvae_lambda_tc),
        regularizer_warmup_epochs=int(args.regularizer_warmup_epochs),
        k=int(args.k),
        epsilon=float(args.epsilon),
        iters=int(args.iters),
        bootstrap_samples=int(args.bootstrap_samples),
        permutation_samples=int(args.permutation_samples),
        minority_quantile=float(args.minority_quantile),
        prefer_cuda=bool(args.prefer_cuda),
    )
    tiny = {
        "three_factor_necessity_checks": out["three_factor_necessity_checks"],
        "variants": list(out["variant_summary"].keys()),
    }
    print(json.dumps(tiny, ensure_ascii=False))


if __name__ == "__main__":
    main()
