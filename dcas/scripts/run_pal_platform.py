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
    constraint_info = build_constraints_from_annotations(
        annotations_csv=str(cfg["annotations_csv"]),
        out_path=str(constraints_path),
    )

    model_path = Path(str(cfg["model_dir"])) / "real_pal_model.pt"
    train_info = train_model(
        tracks_path=str(cfg["tracks"]),
        out_path=model_path,
        constraints_path=str(constraints_path),
        epochs=int(cfg.get("epochs", 8)),
        batch_size=int(cfg.get("batch_size", 128)),
        lr=float(cfg.get("lr", 2e-3)),
        seed=int(cfg.get("seed", 42)),
        prefer_cuda=bool(cfg.get("prefer_cuda", False)),
        lambda_constraints=float(cfg.get("lambda_constraints", 0.2)),
        constraint_margin=float(cfg.get("constraint_margin", 1.0)),
        lambda_domain=float(cfg.get("lambda_domain", 0.5)),
        lambda_contrast=float(cfg.get("lambda_contrast", 0.2)),
        lambda_cov=float(cfg.get("lambda_cov", 0.05)),
    )

    baseline_eval = out_dir / "baseline_eval.json"
    real_eval = out_dir / "real_pal_eval.json"
    evaluate_recommender(
        model_path=str(cfg["baseline_model"]),
        tracks_path=str(cfg["tracks"]),
        interactions_path=str(cfg["interactions"]),
        out_json=str(baseline_eval),
        method="ot",
        k=int(cfg.get("k", 20)),
        epsilon=float(cfg.get("epsilon", 0.1)),
        iters=int(cfg.get("iters", 200)),
        prefer_cuda=bool(cfg.get("prefer_cuda", False)),
    )
    evaluate_recommender(
        model_path=str(model_path),
        tracks_path=str(cfg["tracks"]),
        interactions_path=str(cfg["interactions"]),
        out_json=str(real_eval),
        method="ot",
        k=int(cfg.get("k", 20)),
        epsilon=float(cfg.get("epsilon", 0.1)),
        iters=int(cfg.get("iters", 200)),
        prefer_cuda=bool(cfg.get("prefer_cuda", False)),
    )
    compare_json = out_dir / "compare_baseline_vs_real_pal.json"
    compare_md = out_dir / "compare_baseline_vs_real_pal.md"
    cmp = compare_recommender_runs(
        base_eval_path=str(baseline_eval),
        candidate_eval_path=str(real_eval),
        metrics=["serendipity", "cultural_calibration_kl", "minority_exposure_at_k"],
        bootstrap_samples=int(cfg.get("bootstrap_samples", 300)),
        permutation_samples=int(cfg.get("permutation_samples", 300)),
        seed=int(cfg.get("seed", 42)),
        out_json=str(compare_json),
        out_md=str(compare_md),
    )
    return {
        "mode": "real_round",
        "constraint_info": constraint_info,
        "train_info": {"checkpoint": str(model_path), "history_tail": train_info["history"][-3:]},
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
