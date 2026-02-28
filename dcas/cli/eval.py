from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dcas.scripts.compare_recommender_runs import compare_recommender_runs
from dcas.scripts.evaluate_disentanglement import evaluate_disentanglement
from dcas.scripts.evaluate_recommender import evaluate_recommender


def _parse_csv(raw: str) -> list[str]:
    return [x.strip() for x in str(raw).split(",") if x.strip()]


def _parse_seeds(raw: str) -> list[int]:
    out = [int(x.strip()) for x in str(raw).split(",") if x.strip()]
    if not out:
        return [42]
    return out


def run_eval_suite(
    model_path: str,
    tracks_path: str,
    out_dir: str,
    interactions_path: str | None = None,
    baseline_model_path: str | None = None,
    metadata_csv: str | None = None,
    factors: list[str] | None = None,
    method: str = "ot",
    k: int = 10,
    epsilon: float = 0.1,
    iters: int = 200,
    bootstrap_samples: int = 5000,
    permutation_samples: int = 5000,
    seeds: list[int] | None = None,
    n_bins: int = 20,
    test_ratio: float = 0.2,
    probe_epochs: int = 200,
    probe_lr: float = 1e-2,
    probe_weight_decay: float = 1e-4,
    prefer_cuda: bool = False,
) -> dict[str, Any]:
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)

    suite: dict[str, Any] = {
        "config": {
            "model_path": str(model_path),
            "tracks_path": str(tracks_path),
            "interactions_path": str(interactions_path) if interactions_path else None,
            "baseline_model_path": str(baseline_model_path) if baseline_model_path else None,
            "metadata_csv": str(metadata_csv) if metadata_csv else None,
            "factors": factors or [],
            "method": str(method),
            "k": int(k),
            "epsilon": float(epsilon),
            "iters": int(iters),
            "bootstrap_samples": int(bootstrap_samples),
            "permutation_samples": int(permutation_samples),
            "seeds": list(seeds or [42]),
            "n_bins": int(n_bins),
            "test_ratio": float(test_ratio),
            "probe_epochs": int(probe_epochs),
            "probe_lr": float(probe_lr),
            "probe_weight_decay": float(probe_weight_decay),
            "prefer_cuda": bool(prefer_cuda),
        },
        "artifacts": {},
    }

    if interactions_path:
        rec_json = out_dir_p / "recommender_eval.json"
        rec = evaluate_recommender(
            model_path=str(model_path),
            tracks_path=str(tracks_path),
            interactions_path=str(interactions_path),
            out_json=rec_json,
            method=str(method),
            k=int(k),
            epsilon=float(epsilon),
            iters=int(iters),
            bootstrap_samples=int(bootstrap_samples),
            bootstrap_seed=42,
            prefer_cuda=bool(prefer_cuda),
        )
        suite["artifacts"]["recommender_eval_json"] = str(rec_json)
        suite["recommender_summary"] = rec["summary"]

        if baseline_model_path:
            base_json = out_dir_p / "recommender_eval_baseline.json"
            evaluate_recommender(
                model_path=str(baseline_model_path),
                tracks_path=str(tracks_path),
                interactions_path=str(interactions_path),
                out_json=base_json,
                method=str(method),
                k=int(k),
                epsilon=float(epsilon),
                iters=int(iters),
                bootstrap_samples=int(bootstrap_samples),
                bootstrap_seed=43,
                prefer_cuda=bool(prefer_cuda),
            )
            cmp_json = out_dir_p / "recommender_compare.json"
            cmp_md = out_dir_p / "recommender_compare.md"
            cmp = compare_recommender_runs(
                base_eval_path=base_json,
                candidate_eval_path=rec_json,
                metrics=["serendipity", "cultural_calibration_kl", "minority_exposure_at_k", "target_culture_prob_mean"],
                bootstrap_samples=int(bootstrap_samples),
                permutation_samples=int(permutation_samples),
                seed=44,
                out_json=cmp_json,
                out_md=cmp_md,
            )
            suite["artifacts"]["recommender_baseline_eval_json"] = str(base_json)
            suite["artifacts"]["recommender_compare_json"] = str(cmp_json)
            suite["artifacts"]["recommender_compare_md"] = str(cmp_md)
            suite["recommender_compare_summary"] = cmp

    if metadata_csv and factors:
        dis_json = out_dir_p / "disentanglement_eval.json"
        dis_md = out_dir_p / "disentanglement_eval.md"
        dis = evaluate_disentanglement(
            model_path=str(model_path),
            tracks_path=str(tracks_path),
            metadata_csv=str(metadata_csv),
            factor_cols=list(factors),
            out_json=dis_json,
            out_md=dis_md,
            seeds=list(seeds or [42]),
            n_bins=int(n_bins),
            test_ratio=float(test_ratio),
            probe_epochs=int(probe_epochs),
            probe_lr=float(probe_lr),
            probe_weight_decay=float(probe_weight_decay),
            prefer_cuda=bool(prefer_cuda),
        )
        suite["artifacts"]["disentanglement_eval_json"] = str(dis_json)
        suite["artifacts"]["disentanglement_eval_md"] = str(dis_md)
        suite["disentanglement_summary"] = {
            "n_samples_used": int(dis["n_samples_used"]),
            "spaces": {
                name: {
                    "MIG_mean": float(space["summary"]["MIG_mean"]),
                    "DCI_disentanglement_mean": float(space["summary"]["DCI_disentanglement_mean"]),
                    "DCI_completeness_mean": float(space["summary"]["DCI_completeness_mean"]),
                    "DCI_informativeness_mean": float(space["summary"]["DCI_informativeness_mean"]),
                    "SAP_mean": float(space["summary"]["SAP_mean"]),
                }
                for name, space in dis["spaces"].items()
            },
        }

    suite_json = out_dir_p / "eval_suite_summary.json"
    with open(suite_json, "w", encoding="utf-8") as f:
        json.dump(suite, f, ensure_ascii=False, indent=2)
    suite["artifacts"]["eval_suite_summary_json"] = str(suite_json)
    return suite


def main() -> None:
    ap = argparse.ArgumentParser(description="Unified eval suite for recommender + disentanglement + significance.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--interactions", default=None)
    ap.add_argument("--baseline_model", default=None)

    ap.add_argument("--metadata", default=None)
    ap.add_argument("--factors", default="culture,label")
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--n_bins", type=int, default=20)
    ap.add_argument("--test_ratio", type=float, default=0.2)
    ap.add_argument("--probe_epochs", type=int, default=200)
    ap.add_argument("--probe_lr", type=float, default=1e-2)
    ap.add_argument("--probe_weight_decay", type=float, default=1e-4)

    ap.add_argument("--method", default="ot", choices=["ot", "knn"])
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--epsilon", type=float, default=0.1)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--bootstrap_samples", type=int, default=5000)
    ap.add_argument("--permutation_samples", type=int, default=5000)
    ap.add_argument("--prefer_cuda", action="store_true")
    args = ap.parse_args()

    factors = _parse_csv(str(args.factors))
    seeds = _parse_seeds(str(args.seeds))

    out = run_eval_suite(
        model_path=str(args.model),
        tracks_path=str(args.tracks),
        out_dir=str(args.out_dir),
        interactions_path=str(args.interactions) if args.interactions else None,
        baseline_model_path=str(args.baseline_model) if args.baseline_model else None,
        metadata_csv=str(args.metadata) if args.metadata else None,
        factors=factors,
        method=str(args.method),
        k=int(args.k),
        epsilon=float(args.epsilon),
        iters=int(args.iters),
        bootstrap_samples=int(args.bootstrap_samples),
        permutation_samples=int(args.permutation_samples),
        seeds=seeds,
        n_bins=int(args.n_bins),
        test_ratio=float(args.test_ratio),
        probe_epochs=int(args.probe_epochs),
        probe_lr=float(args.probe_lr),
        probe_weight_decay=float(args.probe_weight_decay),
        prefer_cuda=bool(args.prefer_cuda),
    )

    tiny = {
        "artifacts": out.get("artifacts", {}),
        "recommender_summary": out.get("recommender_summary"),
        "disentanglement_summary": out.get("disentanglement_summary"),
    }
    print(json.dumps(tiny, ensure_ascii=False))


if __name__ == "__main__":
    main()
