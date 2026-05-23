from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dcas.pipelines import pal_tasks, train_model
from dcas.scripts.build_pal_constraints_from_annotations import build_constraints_from_annotations
from dcas.scripts.compare_recommender_runs import compare_recommender_runs
from dcas.scripts.evaluate_recommender import evaluate_recommender
from dcas.scripts.export_pal_annotation_sheet import export_pal_annotation_sheet
from dcas.scripts.run_phase3_pal import run_phase3_pal


def _read_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _tasks_only(cfg: dict[str, Any]) -> dict[str, Any]:
    tasks_path = str(cfg["tasks_path"])
    sheet_path = str(cfg["annotation_sheet_path"])
    task_info = pal_tasks(
        model_path=str(cfg["baseline_model"]),
        tracks_path=str(cfg["tracks"]),
        out_path=tasks_path,
        n=int(cfg.get("tasks_per_round", 120)),
        prefer_cuda=bool(cfg.get("prefer_cuda", False)),
        uncertainty_method=str(cfg.get("uncertainty_method", "auto")),
    )
    sheet_info = export_pal_annotation_sheet(
        tasks_path=tasks_path,
        metadata_csv=str(cfg["metadata"]),
        out_csv=sheet_path,
    )
    return {"mode": "tasks_only", "task_info": task_info, "sheet_info": sheet_info}


def _simulate_rounds(cfg: dict[str, Any]) -> dict[str, Any]:
    return run_phase3_pal(
        tracks_path=str(cfg["tracks"]),
        interactions_path=str(cfg["interactions"]),
        metadata_csv=str(cfg["metadata"]),
        baseline_model_path=str(cfg["baseline_model"]),
        out_dir=str(cfg["out_dir"]),
        artifacts_dir=str(cfg["artifacts_dir"]),
        model_dir=str(cfg["model_dir"]),
        rounds=int(cfg.get("rounds", 2)),
        tasks_per_round=int(cfg.get("tasks_per_round", 120)),
        label_col=str(cfg.get("label_col", "label")),
        uncertainty_method=str(cfg.get("uncertainty_method", "auto")),
        epochs=int(cfg.get("epochs", 8)),
        batch_size=int(cfg.get("batch_size", 128)),
        lr=float(cfg.get("lr", 2e-3)),
        seed=int(cfg.get("seed", 42)),
        lambda_constraints=float(cfg.get("lambda_constraints", 0.2)),
        constraint_margin=float(cfg.get("constraint_margin", 1.0)),
        lambda_domain=float(cfg.get("lambda_domain", 0.5)),
        lambda_contrast=float(cfg.get("lambda_contrast", 0.2)),
        lambda_cov=float(cfg.get("lambda_cov", 0.05)),
        k=int(cfg.get("k", 20)),
        epsilon=float(cfg.get("epsilon", 0.1)),
        iters=int(cfg.get("iters", 200)),
        bootstrap_samples=int(cfg.get("bootstrap_samples", 300)),
        permutation_samples=int(cfg.get("permutation_samples", 300)),
        prefer_cuda=bool(cfg.get("prefer_cuda", False)),
    )


def _real_round(cfg: dict[str, Any]) -> dict[str, Any]:
    out_dir = Path(str(cfg["out_dir"]))
    out_dir.mkdir(parents=True, exist_ok=True)
    constraints_path = out_dir / "real_constraints.jsonl"
    constraints_report_path = out_dir / "real_constraints_report.json"
    constraint_info = build_constraints_from_annotations(
        annotations_csv=str(cfg["annotations_csv"]),
        out_path=str(constraints_path),
        conflict_policy=str(cfg.get("annotation_conflict_policy", "last")),
        report_path=str(constraints_report_path),
    )

    model_path = Path(str(cfg["model_dir"])) / "real_pal_model.pt"
    init_checkpoint_path = str(cfg["init_checkpoint"]) if cfg.get("init_checkpoint") else None
    if init_checkpoint_path is None and bool(cfg.get("warm_start_from_baseline", False)):
        init_checkpoint_path = str(cfg["baseline_model"])
    train_info = train_model(
        tracks_path=str(cfg["tracks"]),
        out_path=model_path,
        constraints_path=str(constraints_path),
        init_checkpoint_path=init_checkpoint_path,
        strict_init=bool(cfg.get("strict_warm_start", True)),
        epochs=int(cfg.get("epochs", 8)),
        batch_size=int(cfg.get("batch_size", 128)),
        lr=float(cfg.get("lr", 2e-3)),
        seed=int(cfg.get("seed", 42)),
        prefer_cuda=bool(cfg.get("prefer_cuda", False)),
        lambda_constraints=float(cfg.get("lambda_constraints", 0.2)),
        constraint_margin=float(cfg.get("constraint_margin", 1.0)),
        constraint_batch_size=int(cfg.get("constraint_batch_size", 64)),
        constraint_candidate_pool_size=int(cfg.get("constraint_candidate_pool_size", 256)),
        constraint_hard_mining=bool(cfg.get("constraint_hard_mining", False)),
        constraint_start_epoch=int(cfg.get("constraint_start_epoch", 0)),
        constraint_warmup_epochs=int(cfg.get("constraint_warmup_epochs", 0)),
        interactions_path=(
            str(cfg["interactions"])
            if cfg.get("interactions") and (bool(cfg.get("preserve_ranking_signal", False)) or float(cfg.get("lambda_rank", 0.0)) > 0.0)
            else None
        ),
        lambda_rank=float(cfg.get("lambda_rank", 0.0)),
        ranking_batch_size=int(cfg.get("ranking_batch_size", 32)),
        ranking_negatives=int(cfg.get("ranking_negatives", 4)),
        ranking_margin=float(cfg.get("ranking_margin", 0.2)),
        ranking_same_culture_ratio=float(cfg.get("ranking_same_culture_ratio", 0.5)),
        rank_start_epoch=int(cfg.get("rank_start_epoch", 0)),
        rank_warmup_epochs=int(cfg.get("rank_warmup_epochs", 0)),
        lambda_domain=float(cfg.get("lambda_domain", 0.5)),
        lambda_contrast=float(cfg.get("lambda_contrast", 0.2)),
        lambda_cov=float(cfg.get("lambda_cov", 0.05)),
        lambda_tc=float(cfg.get("lambda_tc", 0.05)),
        lambda_hsic=float(cfg.get("lambda_hsic", 0.02)),
        lambda_source=float(cfg.get("lambda_source", 0.0)),
        beta_kl=float(cfg.get("beta_kl", 1.0)),
        shared_encoder=bool(cfg.get("shared_encoder", False)),
        regularizer_warmup_epochs=int(cfg.get("regularizer_warmup_epochs", 0)),
        source_balanced_batch=bool(cfg.get("source_balanced_batch", False)),
    )

    baseline_eval = out_dir / "baseline_eval.json"
    real_eval = out_dir / "real_pal_eval.json"
    eval_method = str(cfg.get("eval_method", "ot"))
    evaluate_recommender(
        model_path=str(cfg["baseline_model"]),
        tracks_path=str(cfg["tracks"]),
        interactions_path=str(cfg["interactions"]),
        out_json=str(baseline_eval),
        method=str(eval_method),
        k=int(cfg.get("k", 20)),
        epsilon=float(cfg.get("epsilon", 0.1)),
        iters=int(cfg.get("iters", 200)),
        relevance_weight=float(cfg.get("relevance_weight", 0.62)),
        novelty_weight=float(cfg.get("novelty_weight", 0.12)),
        target_affinity_weight=float(cfg.get("target_affinity_weight", 0.14)),
        minority_weight=float(cfg.get("minority_weight", 0.08)),
        source_weight=float(cfg.get("source_weight", 0.04)),
        diversity_lambda=float(cfg.get("diversity_lambda", 0.03)),
        prefer_cuda=bool(cfg.get("prefer_cuda", False)),
    )
    evaluate_recommender(
        model_path=str(model_path),
        tracks_path=str(cfg["tracks"]),
        interactions_path=str(cfg["interactions"]),
        out_json=str(real_eval),
        method=str(eval_method),
        k=int(cfg.get("k", 20)),
        epsilon=float(cfg.get("epsilon", 0.1)),
        iters=int(cfg.get("iters", 200)),
        relevance_weight=float(cfg.get("relevance_weight", 0.62)),
        novelty_weight=float(cfg.get("novelty_weight", 0.12)),
        target_affinity_weight=float(cfg.get("target_affinity_weight", 0.14)),
        minority_weight=float(cfg.get("minority_weight", 0.08)),
        source_weight=float(cfg.get("source_weight", 0.04)),
        diversity_lambda=float(cfg.get("diversity_lambda", 0.03)),
        prefer_cuda=bool(cfg.get("prefer_cuda", False)),
    )
    compare_json = out_dir / "compare_baseline_vs_real_pal.json"
    compare_md = out_dir / "compare_baseline_vs_real_pal.md"
    cmp = compare_recommender_runs(
        base_eval_path=str(baseline_eval),
        candidate_eval_path=str(real_eval),
        metrics=list(
            cfg.get(
                "compare_metrics",
                ["serendipity", "cultural_calibration_kl", "minority_exposure_at_k"],
            )
        ),
        bootstrap_samples=int(cfg.get("bootstrap_samples", 300)),
        permutation_samples=int(cfg.get("permutation_samples", 300)),
        seed=int(cfg.get("seed", 42)),
        out_json=str(compare_json),
        out_md=str(compare_md),
    )
    return {
        "mode": "real_round",
        "constraint_info": constraint_info,
        "train_info": {
            "checkpoint": str(model_path),
            "history_tail": train_info["history"][-3:],
            "warm_start": train_info.get("warm_start"),
            "n_constraints": train_info.get("n_constraints"),
            "n_rank_examples": train_info.get("n_rank_examples"),
        },
        "eval_method": str(eval_method),
        "compare": cmp,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Unified PAL platform runner.")
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = _read_config(str(args.config))
    mode = str(cfg.get("mode", "tasks_only"))
    if mode == "tasks_only":
        out = _tasks_only(cfg)
    elif mode == "simulate_rounds":
        out = _simulate_rounds(cfg)
    elif mode == "real_round":
        out = _real_round(cfg)
    else:
        raise ValueError(f"unsupported PAL mode: {mode}")
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
