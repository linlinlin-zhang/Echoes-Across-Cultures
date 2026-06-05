from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dcas.data.interactions import Interaction
from dcas.data.npz_tracks import load_tracks
from dcas.embedding_recommenders import (
    load_bpr_mf,
    load_bpr_listwise_hybrid_ranker,
    load_bpr_tree_hybrid_ranker,
    load_bpr_two_stage_hybrid_ranker,
    recommend_bpr_mf,
    recommend_embedding_bpr_listwise_hybrid,
    recommend_embedding_bpr_tree_hybrid,
    recommend_embedding_bpr_two_stage_hybrid,
    recommend_embedding_cosine,
    recommend_embedding_knn,
    recommend_popularity,
    train_bpr_mf,
    train_bpr_listwise_hybrid_ranker,
    train_bpr_tree_hybrid_ranker,
    train_bpr_two_stage_hybrid_ranker,
)
from dcas.pipelines import train_model
from dcas.recommender import Recommendation, recommend_ot
from dcas.scripts.compare_recommender_runs import compare_recommender_runs
from dcas.serialization import load_checkpoint


@dataclass(frozen=True)
class EvalCase:
    user_id: str
    target_track_id: str


def _read_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _load_interaction_rows(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            user_id = str(row.get("user_id", "")).strip()
            track_id = str(row.get("track_id", "")).strip()
            if not user_id or not track_id:
                continue
            rows.append(
                {
                    "user_id": user_id,
                    "track_id": track_id,
                    "weight": max(1e-3, _safe_float(row.get("weight", 1.0), 1.0)),
                    "timestamp": _safe_float(row.get("timestamp", idx), idx),
                    "row_index": int(idx),
                }
            )
    return rows


def _split_train_eval(
    rows: list[dict[str, Any]],
    min_train_interactions: int,
    holdout_per_user: int = 1,
) -> tuple[list[Interaction], list[EvalCase], dict[str, Any]]:
    by_user: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_user.setdefault(str(row["user_id"]), []).append(row)

    train_rows: list[dict[str, Any]] = []
    eval_cases: list[EvalCase] = []
    dropped_users: list[str] = []
    for user_id, user_rows in sorted(by_user.items()):
        ordered = sorted(user_rows, key=lambda r: (float(r["timestamp"]), int(r["row_index"])))
        if len(ordered) < int(min_train_interactions) + int(holdout_per_user):
            dropped_users.append(str(user_id))
            continue
        holdout = ordered[-int(holdout_per_user) :]
        train_part = ordered[: -int(holdout_per_user)]
        train_rows.extend(train_part)
        for row in holdout:
            eval_cases.append(EvalCase(user_id=str(user_id), target_track_id=str(row["track_id"])))

    interactions = [
        Interaction(
            user_id=str(row["user_id"]),
            track_id=str(row["track_id"]),
            weight=float(row["weight"]),
        )
        for row in train_rows
    ]
    split_report = {
        "n_input_rows": int(len(rows)),
        "n_train_rows": int(len(train_rows)),
        "n_eval_cases": int(len(eval_cases)),
        "n_users_total": int(len(by_user)),
        "n_users_eval": int(len({case.user_id for case in eval_cases})),
        "n_users_dropped": int(len(dropped_users)),
        "min_train_interactions": int(min_train_interactions),
        "holdout_per_user": int(holdout_per_user),
        "dropped_users_preview": dropped_users[:20],
    }
    return interactions, eval_cases, split_report


def _ranking_metrics(rank: int | None, ks: list[int]) -> dict[str, float]:
    out: dict[str, float] = {}
    for k in ks:
        hit = bool(rank is not None and int(rank) <= int(k))
        out[f"recall_at_{k}"] = 1.0 if hit else 0.0
        out[f"hit_rate_at_{k}"] = 1.0 if hit else 0.0
        out[f"ndcg_at_{k}"] = float(1.0 / math.log2(int(rank) + 1.0)) if hit and rank is not None else 0.0
        out[f"mrr_at_{k}"] = float(1.0 / int(rank)) if hit and rank is not None else 0.0
    return out


def _method_summary(rows: list[dict[str, Any]], ks: list[int]) -> dict[str, float]:
    if not rows:
        return {}
    summary: dict[str, float] = {"n_eval_cases": float(len(rows))}
    for k in ks:
        for metric in ("recall", "hit_rate", "ndcg", "mrr"):
            key = f"{metric}_at_{k}"
            values = [float(row[key]) for row in rows]
            summary[f"{key}_mean"] = float(np.mean(values)) if values else float("nan")
        unique_items = {str(item) for row in rows for item in row.get(f"topk_items_at_{k}", [])}
        summary[f"coverage_at_{k}"] = float(len(unique_items)) if unique_items else 0.0
    return summary


def _write_train_interactions_csv(path: Path, interactions: list[Interaction]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["user_id", "track_id", "weight"])
        writer.writeheader()
        for row in interactions:
            writer.writerow(
                {
                    "user_id": str(row.user_id),
                    "track_id": str(row.track_id),
                    "weight": float(row.weight),
                }
            )


def _evaluate_method(
    name: str,
    eval_cases: list[EvalCase],
    recommend_fn: Callable[[str, int], list[Recommendation]],
    ks: list[int],
    out_json: Path,
) -> dict[str, Any]:
    max_k = int(max(ks))
    rows: list[dict[str, Any]] = []
    for case in eval_cases:
        recs = recommend_fn(str(case.user_id), int(max_k))
        rank: int | None = None
        for idx, rec in enumerate(recs, start=1):
            if str(rec.track_id) == str(case.target_track_id):
                rank = int(idx)
                break
        row = {
            "user_id": str(case.user_id),
            "target_culture": "global",
            "holdout_track_id": str(case.target_track_id),
            "target_rank": int(rank) if rank is not None else None,
        }
        row.update(_ranking_metrics(rank=rank, ks=ks))
        for k in ks:
            row[f"topk_items_at_{k}"] = [str(rec.track_id) for rec in recs[: int(k)]]
        rows.append(row)

    obj = {
        "method_name": str(name),
        "summary": _method_summary(rows=rows, ks=ks),
        "rows": rows,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    return obj


def run_log_benchmark_suite(config_path: str | Path) -> dict[str, Any]:
    cfg = _read_config(config_path)
    suite_name = str(cfg.get("suite_name", "log_benchmark"))
    tracks_path = Path(str(cfg["tracks"]))
    interactions_path = Path(str(cfg["interactions"]))
    out_dir = Path(str(cfg["output_dir"]))
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_dir = out_dir / "eval"
    cmp_dir = out_dir / "comparisons"
    eval_dir.mkdir(parents=True, exist_ok=True)
    cmp_dir.mkdir(parents=True, exist_ok=True)

    tracks = load_tracks(str(tracks_path))
    raw_rows = _load_interaction_rows(interactions_path)
    train_interactions, eval_cases, split_report = _split_train_eval(
        rows=raw_rows,
        min_train_interactions=int(cfg.get("min_train_interactions", 10)),
        holdout_per_user=int(cfg.get("holdout_per_user", 1)),
    )
    if not eval_cases:
        raise RuntimeError("no eval cases after split")

    train_interactions_csv = out_dir / "_train_interactions.csv"
    _write_train_interactions_csv(train_interactions_csv, train_interactions)

    prefer_cuda = bool(cfg.get("prefer_cuda", False))
    ks = sorted({int(x) for x in cfg.get("k_values", [10, 20])})
    metric_names = [
        *(f"recall_at_{k}" for k in ks),
        *(f"ndcg_at_{k}" for k in ks),
        *(f"mrr_at_{k}" for k in ks),
    ]

    bpr_cfg = dict(cfg.get("bpr", {}))
    bpr_hybrid_cfg = dict(cfg.get("bpr_hybrid", {}))
    bpr_listwise_hybrid_cfg = dict(cfg.get("bpr_listwise_hybrid", {}))
    bpr_tree_hybrid_cfg = dict(cfg.get("bpr_tree_hybrid", {}))
    dcas_train_cfg = dict(cfg.get("dcas_train", {}))

    results_by_method: dict[str, dict[str, Any]] = {}

    for method_cfg in cfg.get("methods", []):
        family = str(method_cfg.get("family", "raw"))
        kind = str(method_cfg.get("kind", ""))
        name = str(method_cfg["name"])

        if family == "raw":
            if kind == "popularity":

                def recommend(user_id, top_k, _tracks=tracks, _ints=train_interactions):
                    return recommend_popularity(_tracks, _ints, user_id, "global", k=top_k)
            elif kind == "cosine":

                def recommend(user_id, top_k, _tracks=tracks, _ints=train_interactions):
                    return recommend_embedding_cosine(_tracks, _ints, user_id, "global", k=top_k)
            elif kind == "knn":

                def recommend(user_id, top_k, _tracks=tracks, _ints=train_interactions):
                    return recommend_embedding_knn(_tracks, _ints, user_id, "global", k=top_k)
            elif kind == "bpr":
                ckpt = str(bpr_cfg.get("checkpoint", out_dir / "bpr_mf.pt"))
                if not Path(ckpt).exists():
                    train_bpr_mf(
                        tracks=tracks,
                        interactions=train_interactions,
                        out_path=ckpt,
                        latent_dim=int(bpr_cfg.get("latent_dim", 64)),
                        epochs=int(bpr_cfg.get("epochs", 8)),
                        batch_size=int(bpr_cfg.get("batch_size", 512)),
                        lr=float(bpr_cfg.get("lr", 5e-3)),
                        reg=float(bpr_cfg.get("reg", 1e-4)),
                        seed=int(bpr_cfg.get("seed", 42)),
                        prefer_cuda=bool(bpr_cfg.get("prefer_cuda", prefer_cuda)),
                    )
                device = torch.device("cuda" if prefer_cuda and torch.cuda.is_available() else "cpu")
                ranker, user_to_id = load_bpr_mf(ckpt, map_location=str(device))
                ranker.to(device)

                def recommend(
                    user_id,
                    top_k,
                    _tracks=tracks,
                    _ints=train_interactions,
                    _ranker=ranker,
                    _users=user_to_id,
                    _device=device,
                ):
                    return recommend_bpr_mf(
                        _ranker,
                        _users,
                        _tracks,
                        _ints,
                        user_id,
                        "global",
                        k=top_k,
                        device=_device,
                    )
            elif kind == "bpr_two_stage_hybrid":
                bpr_ckpt = str(bpr_cfg.get("checkpoint", out_dir / "bpr_mf.pt"))
                if not Path(bpr_ckpt).exists():
                    train_bpr_mf(
                        tracks=tracks,
                        interactions=train_interactions,
                        out_path=bpr_ckpt,
                        latent_dim=int(bpr_cfg.get("latent_dim", 64)),
                        epochs=int(bpr_cfg.get("epochs", 8)),
                        batch_size=int(bpr_cfg.get("batch_size", 512)),
                        lr=float(bpr_cfg.get("lr", 5e-3)),
                        reg=float(bpr_cfg.get("reg", 1e-4)),
                        seed=int(bpr_cfg.get("seed", 42)),
                        prefer_cuda=bool(bpr_cfg.get("prefer_cuda", prefer_cuda)),
                    )
                ckpt = str(bpr_hybrid_cfg.get("checkpoint", out_dir / "bpr_two_stage_hybrid.pt"))
                if not Path(ckpt).exists():
                    train_bpr_two_stage_hybrid_ranker(
                        tracks=tracks,
                        interactions=train_interactions,
                        bpr_checkpoint=bpr_ckpt,
                        out_path=ckpt,
                        hidden_dim=int(bpr_hybrid_cfg.get("hidden_dim", 128)),
                        depth=int(bpr_hybrid_cfg.get("depth", 3)),
                        dropout=float(bpr_hybrid_cfg.get("dropout", 0.1)),
                        epochs=int(bpr_hybrid_cfg.get("epochs", 4)),
                        batch_size=int(bpr_hybrid_cfg.get("batch_size", 256)),
                        lr=float(bpr_hybrid_cfg.get("lr", 1e-3)),
                        recall_k=int(bpr_hybrid_cfg.get("recall_k", 120)),
                        negative_samples=int(bpr_hybrid_cfg.get("negative_samples", 4)),
                        hard_negative_ratio=float(bpr_hybrid_cfg.get("hard_negative_ratio", 0.75)),
                        seed=int(bpr_hybrid_cfg.get("seed", 42)),
                        prefer_cuda=bool(bpr_hybrid_cfg.get("prefer_cuda", prefer_cuda)),
                    )
                device = torch.device("cuda" if prefer_cuda and torch.cuda.is_available() else "cpu")
                bpr_model, user_to_id = load_bpr_mf(bpr_ckpt, map_location=str(device))
                bpr_model.to(device)
                ranker = load_bpr_two_stage_hybrid_ranker(ckpt, map_location=str(device))

                def recommend(
                    user_id,
                    top_k,
                    _tracks=tracks,
                    _ints=train_interactions,
                    _ranker=ranker,
                    _bpr=bpr_model,
                    _users=user_to_id,
                    _device=device,
                    _cfg=bpr_hybrid_cfg,
                ):
                    return recommend_embedding_bpr_two_stage_hybrid(
                        _ranker,
                        _bpr,
                        _users,
                        _tracks,
                        _ints,
                        user_id,
                        "global",
                        k=top_k,
                        recall_k=int(_cfg.get("recall_k", 120)),
                        rerank_weight=float(_cfg.get("rerank_weight", 0.58)),
                        recall_weight=float(_cfg.get("recall_weight", 0.12)),
                        bpr_weight=float(_cfg.get("bpr_weight", 0.08)),
                        novelty_weight=float(_cfg.get("novelty_weight", 0.02)),
                        target_affinity_weight=0.0,
                        minority_weight=0.0,
                        source_weight=0.0,
                        device=_device,
                    )
            elif kind == "bpr_listwise_hybrid":
                bpr_ckpt = str(bpr_cfg.get("checkpoint", out_dir / "bpr_mf.pt"))
                if not Path(bpr_ckpt).exists():
                    train_bpr_mf(
                        tracks=tracks,
                        interactions=train_interactions,
                        out_path=bpr_ckpt,
                        latent_dim=int(bpr_cfg.get("latent_dim", 64)),
                        epochs=int(bpr_cfg.get("epochs", 8)),
                        batch_size=int(bpr_cfg.get("batch_size", 512)),
                        lr=float(bpr_cfg.get("lr", 5e-3)),
                        reg=float(bpr_cfg.get("reg", 1e-4)),
                        seed=int(bpr_cfg.get("seed", 42)),
                        prefer_cuda=bool(bpr_cfg.get("prefer_cuda", prefer_cuda)),
                    )
                ckpt = str(bpr_listwise_hybrid_cfg.get("checkpoint", out_dir / "bpr_listwise_hybrid.pt"))
                if not Path(ckpt).exists():
                    train_bpr_listwise_hybrid_ranker(
                        tracks=tracks,
                        interactions=train_interactions,
                        bpr_checkpoint=bpr_ckpt,
                        out_path=ckpt,
                        warm_start_checkpoint=str(bpr_listwise_hybrid_cfg.get("warm_start_checkpoint", "")) or None,
                        hidden_dim=int(bpr_listwise_hybrid_cfg.get("hidden_dim", 128)),
                        depth=int(bpr_listwise_hybrid_cfg.get("depth", 3)),
                        dropout=float(bpr_listwise_hybrid_cfg.get("dropout", 0.1)),
                        epochs=int(bpr_listwise_hybrid_cfg.get("epochs", 4)),
                        lr=float(bpr_listwise_hybrid_cfg.get("lr", 5e-4)),
                        recall_k=int(bpr_listwise_hybrid_cfg.get("recall_k", 120)),
                        seed=int(bpr_listwise_hybrid_cfg.get("seed", 42)),
                        prefer_cuda=bool(bpr_listwise_hybrid_cfg.get("prefer_cuda", prefer_cuda)),
                    )
                device = torch.device("cuda" if prefer_cuda and torch.cuda.is_available() else "cpu")
                bpr_model, user_to_id = load_bpr_mf(bpr_ckpt, map_location=str(device))
                bpr_model.to(device)
                ranker = load_bpr_listwise_hybrid_ranker(ckpt, map_location=str(device))

                def recommend(
                    user_id,
                    top_k,
                    _tracks=tracks,
                    _ints=train_interactions,
                    _ranker=ranker,
                    _bpr=bpr_model,
                    _users=user_to_id,
                    _device=device,
                    _cfg=bpr_listwise_hybrid_cfg,
                ):
                    return recommend_embedding_bpr_listwise_hybrid(
                        _ranker,
                        _bpr,
                        _users,
                        _tracks,
                        _ints,
                        user_id,
                        "global",
                        k=top_k,
                        recall_k=int(_cfg.get("recall_k", 120)),
                        rerank_weight=float(_cfg.get("rerank_weight", 0.52)),
                        recall_weight=float(_cfg.get("recall_weight", 0.12)),
                        bpr_weight=float(_cfg.get("bpr_weight", 0.08)),
                        novelty_weight=float(_cfg.get("novelty_weight", 0.02)),
                        target_affinity_weight=0.0,
                        minority_weight=0.0,
                        source_weight=0.0,
                        device=_device,
                    )
            elif kind == "bpr_tree_hybrid":
                bpr_ckpt = str(bpr_cfg.get("checkpoint", out_dir / "bpr_mf.pt"))
                if not Path(bpr_ckpt).exists():
                    train_bpr_mf(
                        tracks=tracks,
                        interactions=train_interactions,
                        out_path=bpr_ckpt,
                        latent_dim=int(bpr_cfg.get("latent_dim", 64)),
                        epochs=int(bpr_cfg.get("epochs", 8)),
                        batch_size=int(bpr_cfg.get("batch_size", 512)),
                        lr=float(bpr_cfg.get("lr", 5e-3)),
                        reg=float(bpr_cfg.get("reg", 1e-4)),
                        seed=int(bpr_cfg.get("seed", 42)),
                        prefer_cuda=bool(bpr_cfg.get("prefer_cuda", prefer_cuda)),
                    )
                ckpt = str(bpr_tree_hybrid_cfg.get("checkpoint", out_dir / "bpr_tree_hybrid.pkl"))
                if not Path(ckpt).exists():
                    train_bpr_tree_hybrid_ranker(
                        tracks=tracks,
                        interactions=train_interactions,
                        bpr_checkpoint=bpr_ckpt,
                        out_path=ckpt,
                        backend=str(bpr_tree_hybrid_cfg.get("backend", "lightgbm")),
                        recall_k=int(bpr_tree_hybrid_cfg.get("recall_k", 120)),
                        n_estimators=int(bpr_tree_hybrid_cfg.get("n_estimators", 240)),
                        learning_rate=float(bpr_tree_hybrid_cfg.get("learning_rate", 0.05)),
                        num_leaves=int(bpr_tree_hybrid_cfg.get("num_leaves", 31)),
                        max_depth=int(bpr_tree_hybrid_cfg.get("max_depth", -1)),
                        min_child_samples=int(bpr_tree_hybrid_cfg.get("min_child_samples", 20)),
                        subsample=float(bpr_tree_hybrid_cfg.get("subsample", 0.9)),
                        colsample_bytree=float(bpr_tree_hybrid_cfg.get("colsample_bytree", 0.8)),
                        reg_lambda=float(bpr_tree_hybrid_cfg.get("reg_lambda", 1.0)),
                        seed=int(bpr_tree_hybrid_cfg.get("seed", 42)),
                        prefer_cuda=bool(bpr_tree_hybrid_cfg.get("prefer_cuda", prefer_cuda)),
                    )
                device = torch.device("cuda" if prefer_cuda and torch.cuda.is_available() else "cpu")
                bpr_model, user_to_id = load_bpr_mf(bpr_ckpt, map_location=str(device))
                bpr_model.to(device)
                ranker = load_bpr_tree_hybrid_ranker(ckpt)

                def recommend(
                    user_id,
                    top_k,
                    _tracks=tracks,
                    _ints=train_interactions,
                    _ranker=ranker,
                    _bpr=bpr_model,
                    _users=user_to_id,
                    _device=device,
                    _cfg=bpr_tree_hybrid_cfg,
                ):
                    return recommend_embedding_bpr_tree_hybrid(
                        _ranker,
                        _bpr,
                        _users,
                        _tracks,
                        _ints,
                        user_id,
                        "global",
                        k=top_k,
                        recall_k=int(_cfg.get("recall_k", 120)),
                        rerank_weight=float(_cfg.get("rerank_weight", 0.54)),
                        recall_weight=float(_cfg.get("recall_weight", 0.12)),
                        bpr_weight=float(_cfg.get("bpr_weight", 0.08)),
                        novelty_weight=float(_cfg.get("novelty_weight", 0.02)),
                        target_affinity_weight=0.0,
                        minority_weight=0.0,
                        source_weight=0.0,
                        device=_device,
                    )
            else:
                raise ValueError(f"unsupported raw method kind: {kind}")
        elif family == "dcas":
            ckpt = str(
                method_cfg.get(
                    "checkpoint",
                    dcas_train_cfg.get("checkpoint", out_dir / "dcas_log_model.pt"),
                )
            )
            if not Path(ckpt).exists():
                train_model(
                    tracks_path=str(tracks_path),
                    out_path=ckpt,
                    interactions_path=str(train_interactions_csv)
                    if float(dcas_train_cfg.get("lambda_rank", 0.0)) > 0
                    else None,
                    epochs=int(dcas_train_cfg.get("epochs", 8)),
                    batch_size=int(dcas_train_cfg.get("batch_size", 128)),
                    lr=float(dcas_train_cfg.get("lr", 2e-3)),
                    seed=int(dcas_train_cfg.get("seed", 42)),
                    prefer_cuda=bool(dcas_train_cfg.get("prefer_cuda", prefer_cuda)),
                    lambda_constraints=float(dcas_train_cfg.get("lambda_constraints", 0.0)),
                    constraint_margin=float(dcas_train_cfg.get("constraint_margin", 1.0)),
                    lambda_domain=float(dcas_train_cfg.get("lambda_domain", 0.0)),
                    lambda_contrast=float(dcas_train_cfg.get("lambda_contrast", 0.2)),
                    lambda_cov=float(dcas_train_cfg.get("lambda_cov", 0.05)),
                    lambda_tc=float(dcas_train_cfg.get("lambda_tc", 0.05)),
                    lambda_hsic=float(dcas_train_cfg.get("lambda_hsic", 0.02)),
                    beta_kl=float(dcas_train_cfg.get("beta_kl", 1.0)),
                    shared_encoder=bool(dcas_train_cfg.get("shared_encoder", False)),
                    regularizer_warmup_epochs=int(dcas_train_cfg.get("regularizer_warmup_epochs", 0)),
                    lambda_source=float(dcas_train_cfg.get("lambda_source", 0.0)),
                    source_balanced_batch=bool(dcas_train_cfg.get("source_balanced_batch", False)),
                    lambda_rank=float(dcas_train_cfg.get("lambda_rank", 0.0)),
                    ranking_batch_size=int(dcas_train_cfg.get("ranking_batch_size", 32)),
                    ranking_negatives=int(dcas_train_cfg.get("ranking_negatives", 4)),
                    ranking_margin=float(dcas_train_cfg.get("ranking_margin", 0.2)),
                    ranking_same_culture_ratio=float(dcas_train_cfg.get("ranking_same_culture_ratio", 1.0)),
                    rank_start_epoch=int(dcas_train_cfg.get("rank_start_epoch", 0)),
                    rank_warmup_epochs=int(dcas_train_cfg.get("rank_warmup_epochs", 0)),
                )
            device = torch.device("cuda" if prefer_cuda and torch.cuda.is_available() else "cpu")
            model, _ = load_checkpoint(ckpt, map_location=str(device))
            if kind != "ot":
                raise ValueError(f"unsupported dcas method kind: {kind}")

            def recommend(
                user_id,
                top_k,
                _model=model,
                _tracks=tracks,
                _ints=train_interactions,
                _device=device,
                _cfg=method_cfg,
            ):
                return recommend_ot(
                    _model,
                    _tracks,
                    _ints,
                    user_id,
                    "global",
                    k=top_k,
                    device=_device,
                    epsilon=float(_cfg.get("epsilon", 0.1)),
                    iters=int(_cfg.get("iters", 200)),
                )[0]
        else:
            raise ValueError(f"unsupported family: {family}")

        results_by_method[name] = _evaluate_method(
            name=name,
            eval_cases=eval_cases,
            recommend_fn=recommend,
            ks=ks,
            out_json=eval_dir / f"{name}.json",
        )

    reference_method = str(cfg.get("reference_method", ""))
    comparisons: dict[str, Any] = {}
    if reference_method and (eval_dir / f"{reference_method}.json").exists():
        ref_eval = eval_dir / f"{reference_method}.json"
        for method_name in sorted(results_by_method.keys()):
            if method_name == reference_method:
                continue
            comparisons[method_name] = compare_recommender_runs(
                base_eval_path=str(eval_dir / f"{method_name}.json"),
                candidate_eval_path=str(ref_eval),
                metrics=metric_names,
                bootstrap_samples=int(cfg.get("bootstrap_samples", 500)),
                permutation_samples=int(cfg.get("permutation_samples", 500)),
                seed=int(cfg.get("bootstrap_seed", 42)),
                out_json=str(cmp_dir / f"{method_name}_vs_{reference_method}.json"),
                out_md=str(cmp_dir / f"{method_name}_vs_{reference_method}.md"),
            )

    summary: dict[str, Any] = {
        "suite_name": suite_name,
        "tracks": str(tracks_path),
        "interactions": str(interactions_path),
        "split": split_report,
        "methods": {name: dict(result.get("summary", {})) for name, result in results_by_method.items()},
        "reference_method": reference_method,
        "comparisons_vs_reference": {
            name: {
                metric: {
                    "delta_mean": float(values.get("delta_mean", float("nan"))),
                    "p_value_two_sided": float(values.get("p_value_two_sided", float("nan"))),
                }
                for metric, values in comparison.get("metrics", {}).items()
            }
            for name, comparison in comparisons.items()
        },
    }
    (out_dir / "benchmark_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        f"# Log Benchmark: {suite_name}",
        "",
        f"- tracks: `{tracks_path}`",
        f"- interactions: `{interactions_path}`",
        f"- eval_users: `{split_report['n_users_eval']}`",
        f"- eval_cases: `{split_report['n_eval_cases']}`",
        "",
        "| method | "
        + " | ".join(f"Recall@{k}" for k in ks)
        + " | "
        + " | ".join(f"NDCG@{k}" for k in ks)
        + " | "
        + " | ".join(f"MRR@{k}" for k in ks)
        + " |",
        "|---|" + "|".join("---:" for _ in range(len(ks) * 3)) + "|",
    ]
    for method_name, result in summary["methods"].items():
        cells = [method_name]
        cells.extend(f"{float(result.get(f'recall_at_{k}_mean', float('nan'))):.4f}" for k in ks)
        cells.extend(f"{float(result.get(f'ndcg_at_{k}_mean', float('nan'))):.4f}" for k in ks)
        cells.extend(f"{float(result.get(f'mrr_at_{k}_mean', float('nan'))):.4f}" for k in ks)
        lines.append("| " + " | ".join(cells) + " |")
    if comparisons:
        lines.extend(
            [
                "",
                "## Comparisons vs Reference",
                "",
                "| method | metric | delta_mean(reference - base) | p_value |",
                "|---|---|---:|---:|",
            ]
        )
        for method_name, comparison in summary["comparisons_vs_reference"].items():
            for metric, values in comparison.items():
                lines.append(
                    f"| {method_name} | {metric} | {float(values.get('delta_mean', float('nan'))):+.6f} | {float(values.get('p_value_two_sided', float('nan'))):.6f} |"
                )
    (out_dir / "benchmark_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Run a log-ranking benchmark using repository recommenders.")
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = _read_config(str(args.config))
    summary = run_log_benchmark_suite(str(args.config))
    print(
        json.dumps(
            {
                "suite_name": summary["suite_name"],
                "summary_json": str((Path(str(cfg["output_dir"])) / "benchmark_summary.json").resolve()),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
