from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Callable

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dcas.data.interactions import Interaction, load_interactions
from dcas.data.npz_tracks import Tracks, load_tracks
from dcas.embedding_recommenders import (
    load_bpr_mf,
    load_bpr_listwise_hybrid_ranker,
    load_bpr_two_stage_hybrid_ranker,
    load_content_bpr_mf,
    load_shallow_ranker,
    load_two_stage_hybrid_ranker,
    recommend_bpr_mf,
    recommend_embedding_bpr_listwise_hybrid,
    recommend_embedding_bpr_two_stage_hybrid,
    recommend_content_bpr_mf,
    recommend_embedding_cosine,
    recommend_embedding_hybrid,
    recommend_embedding_knn,
    recommend_embedding_mlp,
    recommend_embedding_two_stage_hybrid,
    recommend_popularity,
    train_bpr_mf,
    train_bpr_listwise_hybrid_ranker,
    train_bpr_two_stage_hybrid_ranker,
    train_content_bpr_mf,
    train_shallow_ranker,
    train_two_stage_hybrid_ranker,
)
from dcas.recommender import (
    Recommendation,
    recommend_knn,
    recommend_knn_calibrated,
    recommend_open_knn,
    recommend_open_ot,
    recommend_ot,
    recommend_ot_calibrated,
)
from dcas.scripts.compare_recommender_runs import compare_recommender_runs
from dcas.scripts.synthesize_interactions import synthesize_interactions
from dcas.serialization import load_checkpoint


def _safe_mean(values: list[float]) -> float:
    return float(mean(values)) if values else float("nan")


def _safe_kl(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = p.astype(np.float64)
    q = q.astype(np.float64)
    p = p / max(eps, float(p.sum()))
    q = q / max(eps, float(q.sum()))
    return float(np.sum(p * (np.log(p + eps) - np.log(q + eps))))


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


def _track_id_to_idx(tracks: Tracks) -> dict[str, int]:
    return {str(tid): int(i) for i, tid in enumerate(tracks.track_id.tolist())}


def _prepare_user_history(
    tracks: Tracks,
    interactions: list[Interaction],
    user_id: str,
) -> tuple[np.ndarray, np.ndarray]:
    track_id_to_idx = _track_id_to_idx(tracks)
    user_hist = [it for it in interactions if str(it.user_id) == str(user_id) and str(it.track_id) in track_id_to_idx]
    if not user_hist:
        raise ValueError(f"no interactions for user_id={user_id}")
    hist_idx = np.array([track_id_to_idx[str(it.track_id)] for it in user_hist], dtype=np.int64)
    hist_w = np.array([float(it.weight) for it in user_hist], dtype=np.float32)
    hist_w = hist_w / max(1e-12, float(hist_w.sum()))
    return hist_idx, hist_w


def _soft_culture_distribution(points: np.ndarray, centroids: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    if points.shape[0] == 0:
        return np.full((centroids.shape[0],), 1.0 / max(1, centroids.shape[0]), dtype=np.float64)
    d = np.linalg.norm(points[:, None, :] - centroids[None, :, :], axis=2)
    z = -d / max(1e-6, float(temperature))
    z = z - np.max(z, axis=1, keepdims=True)
    p = np.exp(z)
    p = p / np.maximum(1e-12, p.sum(axis=1, keepdims=True))
    return p.mean(axis=0).astype(np.float64)


def _culture_centroids(tracks: Tracks) -> tuple[list[str], np.ndarray]:
    names = tracks.cultures()
    centroids: list[np.ndarray] = []
    for name in names:
        idx = tracks.indices_of_cultures([name])
        centroids.append(tracks.embedding[idx].mean(axis=0))
    return names, np.stack(centroids, axis=0).astype(np.float32)


def _track_popularity_by_id(interactions: list[Interaction]) -> dict[str, float]:
    pop: dict[str, float] = {}
    for it in interactions:
        tid = str(it.track_id)
        pop[tid] = float(pop.get(tid, 0.0) + float(it.weight))
    return pop


def _minority_track_set(
    track_ids: np.ndarray,
    pop_by_id: dict[str, float],
    quantile: float,
) -> tuple[set[str], float, float]:
    q = float(np.clip(float(quantile), 0.0, 1.0))
    ids = [str(tid) for tid in track_ids.tolist()]
    if not ids:
        return set(), float("nan"), float("nan")
    pop = np.array([float(pop_by_id.get(tid, 0.0)) for tid in ids], dtype=np.float64)
    if float(np.max(pop) - np.min(pop)) <= 1e-12:
        n = int(len(ids))
        n_minority = max(1, min(n, int(round(float(n) * max(0.0, q)))))
        order = np.argsort(np.array(ids, dtype=object))
        minority = {ids[int(i)] for i in order[:n_minority].tolist()}
        return minority, float(pop[0]), float(len(minority) / max(1, len(ids)))
    threshold = float(np.quantile(pop, q))
    mask = pop <= threshold
    if int(mask.sum()) <= 0:
        mask[np.argmin(pop)] = True
    minority = {tid for tid, m in zip(ids, mask.tolist()) if bool(m)}
    return minority, threshold, float(len(minority) / max(1, len(ids)))


def _minority_exposure(recs: list[Recommendation], minority_tracks: set[str]) -> float:
    if not recs:
        return float("nan")
    hit = sum(1 for r in recs if str(r.track_id) in minority_tracks)
    return float(hit / len(recs))


def _generic_metrics_from_recs(
    tracks: Tracks,
    recs: list[Recommendation],
    hist_idx: np.ndarray,
    target_culture: str,
    culture_names: list[str],
    centroids: np.ndarray,
    minority_tracks: set[str],
) -> dict[str, float]:
    if not recs:
        raise ValueError("recommender returned no recommendations")
    track_id_to_idx = _track_id_to_idx(tracks)
    rec_idx = np.array([track_id_to_idx[str(r.track_id)] for r in recs if str(r.track_id) in track_id_to_idx], dtype=np.int64)
    if rec_idx.size == 0:
        raise ValueError("recommended track ids are not in tracks catalog")

    unexpected = np.array([float(r.unexpectedness) for r in recs], dtype=np.float64)
    relevant = np.array([float(r.relevance) for r in recs], dtype=np.float64)
    unexpected = unexpected / max(1e-12, float(np.max(unexpected)))
    relevant = relevant / max(1e-12, float(np.max(relevant)))
    serendipity = float(np.mean(unexpected * relevant))

    hist_emb = tracks.embedding[hist_idx]
    rec_emb = tracks.embedding[rec_idx]
    rec_soft = _soft_culture_distribution(rec_emb, centroids, temperature=1.0)
    hist_soft = _soft_culture_distribution(hist_emb, centroids, temperature=1.0)

    target_dist = np.full((len(culture_names),), 0.0, dtype=np.float64)
    if str(target_culture) in culture_names:
        smoothing = 0.05
        target_idx = culture_names.index(str(target_culture))
        off = smoothing / float(max(1, len(culture_names) - 1))
        target_dist[:] = off
        target_dist[target_idx] = 1.0 - smoothing
    else:
        target_dist[:] = 1.0 / max(1, len(culture_names))

    target_prob = float(rec_soft[culture_names.index(str(target_culture))]) if str(target_culture) in culture_names else float("nan")
    return {
        "serendipity": serendipity,
        "cultural_calibration_kl": _safe_kl(rec_soft, target_dist),
        "target_culture_prob_mean": target_prob,
        "user_culture_alignment_kl": _safe_kl(rec_soft, hist_soft),
        "minority_exposure_at_k": float(_minority_exposure(recs, minority_tracks)),
    }


def evaluate_callable_recommender(
    name: str,
    tracks: Tracks,
    interactions: list[Interaction],
    recommend_fn: Callable[[str, str, int], list[Recommendation]],
    out_json: str | Path | None = None,
    k: int = 20,
    bootstrap_samples: int = 2000,
    bootstrap_seed: int = 42,
    minority_quantile: float = 0.25,
) -> dict[str, Any]:
    users = sorted({str(i.user_id) for i in interactions})
    cultures = tracks.cultures()
    culture_names, centroids = _culture_centroids(tracks)
    pop_by_id = _track_popularity_by_id(interactions)
    minority_tracks, minority_threshold, minority_ratio = _minority_track_set(
        track_ids=tracks.track_id,
        pop_by_id=pop_by_id,
        quantile=float(minority_quantile),
    )

    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for user_id in users:
        for target_culture in cultures:
            try:
                hist_idx, _ = _prepare_user_history(tracks=tracks, interactions=interactions, user_id=user_id)
                recs = recommend_fn(user_id, target_culture, int(k))
                metrics = _generic_metrics_from_recs(
                    tracks=tracks,
                    recs=recs,
                    hist_idx=hist_idx,
                    target_culture=target_culture,
                    culture_names=culture_names,
                    centroids=centroids,
                    minority_tracks=minority_tracks,
                )
                row = {"user_id": user_id, "target_culture": target_culture, **{k_: float(v) for k_, v in metrics.items()}}
                rows.append(row)
            except Exception as e:
                skipped.append({"user_id": str(user_id), "target_culture": str(target_culture), "reason": str(e)})

    ser = [float(r["serendipity"]) for r in rows]
    ckl = [float(r["cultural_calibration_kl"]) for r in rows]
    target_prob = [float(r["target_culture_prob_mean"]) for r in rows]
    user_align_kl = [float(r["user_culture_alignment_kl"]) for r in rows]
    minority = [float(r["minority_exposure_at_k"]) for r in rows]

    per_culture: dict[str, dict[str, float]] = {}
    tmp: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"ser": [], "ckl": [], "target_prob": [], "user_align_kl": [], "minority": []}
    )
    for r in rows:
        c = str(r["target_culture"])
        tmp[c]["ser"].append(float(r["serendipity"]))
        tmp[c]["ckl"].append(float(r["cultural_calibration_kl"]))
        tmp[c]["target_prob"].append(float(r["target_culture_prob_mean"]))
        tmp[c]["user_align_kl"].append(float(r["user_culture_alignment_kl"]))
        tmp[c]["minority"].append(float(r["minority_exposure_at_k"]))
    for c in sorted(tmp.keys()):
        ser_ci_l, ser_ci_h = _ci95_bootstrap(tmp[c]["ser"], samples=int(bootstrap_samples), seed=int(bootstrap_seed) + 11)
        ckl_ci_l, ckl_ci_h = _ci95_bootstrap(tmp[c]["ckl"], samples=int(bootstrap_samples), seed=int(bootstrap_seed) + 13)
        min_ci_l, min_ci_h = _ci95_bootstrap(tmp[c]["minority"], samples=int(bootstrap_samples), seed=int(bootstrap_seed) + 17)
        per_culture[c] = {
            "n": int(len(tmp[c]["ser"])),
            "serendipity_mean": _safe_mean(tmp[c]["ser"]),
            "serendipity_ci95_low": float(ser_ci_l),
            "serendipity_ci95_high": float(ser_ci_h),
            "cultural_calibration_kl_mean": _safe_mean(tmp[c]["ckl"]),
            "cultural_calibration_kl_ci95_low": float(ckl_ci_l),
            "cultural_calibration_kl_ci95_high": float(ckl_ci_h),
            "target_culture_prob_mean": _safe_mean(tmp[c]["target_prob"]),
            "user_culture_alignment_kl_mean": _safe_mean(tmp[c]["user_align_kl"]),
            "minority_exposure_at_k_mean": _safe_mean(tmp[c]["minority"]),
            "minority_exposure_at_k_ci95_low": float(min_ci_l),
            "minority_exposure_at_k_ci95_high": float(min_ci_h),
        }

    ser_ci_l, ser_ci_h = _ci95_bootstrap(ser, samples=int(bootstrap_samples), seed=int(bootstrap_seed))
    ckl_ci_l, ckl_ci_h = _ci95_bootstrap(ckl, samples=int(bootstrap_samples), seed=int(bootstrap_seed) + 1)
    min_ci_l, min_ci_h = _ci95_bootstrap(minority, samples=int(bootstrap_samples), seed=int(bootstrap_seed) + 2)
    result: dict[str, Any] = {
        "summary": {
            "method_name": str(name),
            "n_users": int(len(users)),
            "n_cultures": int(len(cultures)),
            "n_user_culture_evals": int(len(rows)),
            "n_skipped": int(len(skipped)),
            "serendipity_mean": _safe_mean(ser),
            "serendipity_ci95_low": float(ser_ci_l),
            "serendipity_ci95_high": float(ser_ci_h),
            "cultural_calibration_kl_mean": _safe_mean(ckl),
            "cultural_calibration_kl_ci95_low": float(ckl_ci_l),
            "cultural_calibration_kl_ci95_high": float(ckl_ci_h),
            "target_culture_prob_mean": _safe_mean(target_prob),
            "user_culture_alignment_kl_mean": _safe_mean(user_align_kl),
            "minority_exposure_at_k_mean": _safe_mean(minority),
            "minority_exposure_at_k_ci95_low": float(min_ci_l),
            "minority_exposure_at_k_ci95_high": float(min_ci_h),
        },
        "per_target_culture": per_culture,
        "rows": rows,
        "skipped": skipped[:200],
        "config": {
            "k": int(k),
            "bootstrap_samples": int(bootstrap_samples),
            "bootstrap_seed": int(bootstrap_seed),
            "minority_quantile": float(minority_quantile),
            "minority_popularity_threshold": float(minority_threshold),
            "minority_catalog_ratio": float(minority_ratio),
        },
    }
    if out_json is not None:
        out = Path(out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    return result


def _ensure_interactions(
    interactions_path: str | None,
    metadata_path: str | None,
    synth_cfg: dict[str, Any] | None,
) -> str:
    if interactions_path is not None and Path(interactions_path).exists():
        return str(Path(interactions_path))
    if not synth_cfg or not bool(synth_cfg.get("enabled", False)):
        raise RuntimeError("interactions file is missing and auto_synthesize_interactions is not enabled")
    if metadata_path is None:
        raise RuntimeError("metadata is required for auto_synthesize_interactions")
    out_csv = synth_cfg.get("out")
    if not out_csv:
        raise RuntimeError("auto_synthesize_interactions.out is required")
    synthesize_interactions(
        metadata_csv=metadata_path,
        out_csv=out_csv,
        users_per_culture=int(synth_cfg.get("users_per_culture", 20)),
        tracks_per_user=int(synth_cfg.get("tracks_per_user", 50)),
        min_weight=float(synth_cfg.get("min_weight", 0.5)),
        max_weight=float(synth_cfg.get("max_weight", 2.0)),
        genre_column=str(synth_cfg.get("genre_column", "label")),
        mode=str(synth_cfg.get("mode", "single_culture")),
        secondary_cultures=int(synth_cfg.get("secondary_cultures", 2)),
        home_share=float(synth_cfg.get("home_share", 0.65)),
        seed=int(synth_cfg.get("seed", 42)),
    )
    return str(Path(out_csv))


def run_benchmark_suite(config_path: str | Path) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8-sig") as f:
        cfg = json.load(f)

    suite_name = str(cfg.get("suite_name", Path(config_path).stem))
    tracks_path = str(cfg["tracks"])
    metadata_path = str(cfg["metadata"]) if cfg.get("metadata") else None
    interactions_path = _ensure_interactions(
        interactions_path=str(cfg["interactions"]) if cfg.get("interactions") else None,
        metadata_path=metadata_path,
        synth_cfg=cfg.get("auto_synthesize_interactions"),
    )
    out_dir = Path(str(cfg["output_dir"]))
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_dir = out_dir / "eval"
    cmp_dir = out_dir / "comparisons"
    eval_dir.mkdir(parents=True, exist_ok=True)
    cmp_dir.mkdir(parents=True, exist_ok=True)

    tracks = load_tracks(tracks_path)
    interactions = load_interactions(interactions_path)
    k = int(cfg.get("k", 20))
    bootstrap_samples = int(cfg.get("bootstrap_samples", 2000))
    bootstrap_seed = int(cfg.get("bootstrap_seed", 42))
    minority_quantile = float(cfg.get("minority_quantile", 0.25))
    prefer_cuda = bool(cfg.get("prefer_cuda", False))

    mlp_cfg = dict(cfg.get("mlp", {}))
    hybrid_weights = dict(cfg.get("hybrid_weights", {}))
    strong_hybrid_cfg = dict(cfg.get("strong_hybrid", {}))
    bpr_cfg = dict(cfg.get("bpr", {}))
    bpr_hybrid_cfg = dict(cfg.get("bpr_hybrid", {}))
    bpr_listwise_hybrid_cfg = dict(cfg.get("bpr_listwise_hybrid", {}))
    lightfm_like_cfg = dict(cfg.get("lightfm_like", {}))

    results_by_method: dict[str, dict[str, Any]] = {}
    eval_paths: dict[str, str] = {}

    for method_cfg in cfg.get("methods", []):
        name = str(method_cfg["name"])
        family = str(method_cfg.get("family", "raw"))
        kind = str(method_cfg.get("kind"))

        if family == "raw":
            if kind == "popularity":
                recommend_fn = lambda user_id, target_culture, top_k, _tracks=tracks, _ints=interactions: recommend_popularity(
                    _tracks, _ints, user_id, target_culture, k=top_k
                )
            elif kind == "cosine":
                recommend_fn = lambda user_id, target_culture, top_k, _tracks=tracks, _ints=interactions: recommend_embedding_cosine(
                    _tracks, _ints, user_id, target_culture, k=top_k
                )
            elif kind == "knn":
                recommend_fn = lambda user_id, target_culture, top_k, _tracks=tracks, _ints=interactions: recommend_embedding_knn(
                    _tracks, _ints, user_id, target_culture, k=top_k
                )
            elif kind == "hybrid":
                recommend_fn = lambda user_id, target_culture, top_k, _tracks=tracks, _ints=interactions, _w=hybrid_weights: recommend_embedding_hybrid(
                    _tracks,
                    _ints,
                    user_id,
                    target_culture,
                    k=top_k,
                    cosine_weight=float(_w.get("cosine", 0.4)),
                    knn_weight=float(_w.get("knn", 0.25)),
                    popularity_weight=float(_w.get("popularity", 0.2)),
                    novelty_weight=float(_w.get("novelty", 0.15)),
                )
            elif kind == "mlp":
                ckpt = str(mlp_cfg.get("checkpoint", out_dir / "shallow_ranker.pt"))
                if not Path(ckpt).exists():
                    if not bool(mlp_cfg.get("train_if_missing", True)):
                        raise RuntimeError(f"MLP checkpoint missing and train_if_missing=false: {ckpt}")
                    train_shallow_ranker(
                        tracks=tracks,
                        interactions=interactions,
                        out_path=ckpt,
                        hidden_dim=int(mlp_cfg.get("hidden_dim", 128)),
                        epochs=int(mlp_cfg.get("epochs", 5)),
                        batch_size=int(mlp_cfg.get("batch_size", 256)),
                        lr=float(mlp_cfg.get("lr", 1e-3)),
                        negative_samples=int(mlp_cfg.get("negative_samples", 2)),
                        seed=int(mlp_cfg.get("seed", 42)),
                        prefer_cuda=bool(mlp_cfg.get("prefer_cuda", prefer_cuda)),
                    )
                device = torch.device("cuda" if prefer_cuda and torch.cuda.is_available() else "cpu")
                ranker = load_shallow_ranker(ckpt, map_location=str(device))
                recommend_fn = lambda user_id, target_culture, top_k, _tracks=tracks, _ints=interactions, _ranker=ranker, _device=device: recommend_embedding_mlp(
                    _ranker, _tracks, _ints, user_id, target_culture, k=top_k, device=_device
                )
            elif kind == "two_stage_hybrid":
                ckpt = str(strong_hybrid_cfg.get("checkpoint", out_dir / "two_stage_hybrid_ranker.pt"))
                if not Path(ckpt).exists():
                    if not bool(strong_hybrid_cfg.get("train_if_missing", True)):
                        raise RuntimeError(f"two-stage hybrid checkpoint missing and train_if_missing=false: {ckpt}")
                    train_two_stage_hybrid_ranker(
                        tracks=tracks,
                        interactions=interactions,
                        out_path=ckpt,
                        hidden_dim=int(strong_hybrid_cfg.get("hidden_dim", 128)),
                        depth=int(strong_hybrid_cfg.get("depth", 3)),
                        dropout=float(strong_hybrid_cfg.get("dropout", 0.1)),
                        epochs=int(strong_hybrid_cfg.get("epochs", 6)),
                        batch_size=int(strong_hybrid_cfg.get("batch_size", 256)),
                        lr=float(strong_hybrid_cfg.get("lr", 1e-3)),
                        negative_samples=int(strong_hybrid_cfg.get("negative_samples", 4)),
                        recall_k=int(strong_hybrid_cfg.get("recall_k", max(80, 4 * int(k)))),
                        hard_negative_ratio=float(strong_hybrid_cfg.get("hard_negative_ratio", 0.75)),
                        seed=int(strong_hybrid_cfg.get("seed", 42)),
                        prefer_cuda=bool(strong_hybrid_cfg.get("prefer_cuda", prefer_cuda)),
                    )
                device = torch.device("cuda" if prefer_cuda and torch.cuda.is_available() else "cpu")
                ranker = load_two_stage_hybrid_ranker(ckpt, map_location=str(device))
                recommend_fn = lambda user_id, target_culture, top_k, _tracks=tracks, _ints=interactions, _ranker=ranker, _device=device, _cfg=strong_hybrid_cfg: recommend_embedding_two_stage_hybrid(
                    _ranker,
                    _tracks,
                    _ints,
                    user_id,
                    target_culture,
                    k=top_k,
                    recall_k=int(_cfg.get("recall_k", max(80, 4 * int(top_k)))),
                    blend_weight=float(_cfg.get("blend_weight", 0.25)),
                    device=_device,
                )
            elif kind == "bpr":
                ckpt = str(bpr_cfg.get("checkpoint", out_dir / "bpr_mf.pt"))
                if not Path(ckpt).exists():
                    if not bool(bpr_cfg.get("train_if_missing", True)):
                        raise RuntimeError(f"BPR checkpoint missing and train_if_missing=false: {ckpt}")
                    train_bpr_mf(
                        tracks=tracks,
                        interactions=interactions,
                        out_path=ckpt,
                        latent_dim=int(bpr_cfg.get("latent_dim", 64)),
                        epochs=int(bpr_cfg.get("epochs", 10)),
                        batch_size=int(bpr_cfg.get("batch_size", 512)),
                        lr=float(bpr_cfg.get("lr", 5e-3)),
                        reg=float(bpr_cfg.get("reg", 1e-4)),
                        seed=int(bpr_cfg.get("seed", 42)),
                        prefer_cuda=bool(bpr_cfg.get("prefer_cuda", prefer_cuda)),
                    )
                device = torch.device("cuda" if prefer_cuda and torch.cuda.is_available() else "cpu")
                ranker, user_to_id = load_bpr_mf(ckpt, map_location=str(device))
                recommend_fn = lambda user_id, target_culture, top_k, _tracks=tracks, _ints=interactions, _ranker=ranker, _users=user_to_id, _device=device: recommend_bpr_mf(
                    _ranker,
                    _users,
                    _tracks,
                    _ints,
                    user_id,
                    target_culture,
                    k=top_k,
                    device=_device,
                )
            elif kind == "lightfm_like":
                ckpt = str(lightfm_like_cfg.get("checkpoint", out_dir / "lightfm_like.pt"))
                if not Path(ckpt).exists():
                    if not bool(lightfm_like_cfg.get("train_if_missing", True)):
                        raise RuntimeError(f"LightFM-like checkpoint missing and train_if_missing=false: {ckpt}")
                    train_content_bpr_mf(
                        tracks=tracks,
                        interactions=interactions,
                        out_path=ckpt,
                        latent_dim=int(lightfm_like_cfg.get("latent_dim", 64)),
                        epochs=int(lightfm_like_cfg.get("epochs", 12)),
                        batch_size=int(lightfm_like_cfg.get("batch_size", 512)),
                        lr=float(lightfm_like_cfg.get("lr", 3e-3)),
                        reg=float(lightfm_like_cfg.get("reg", 1e-4)),
                        profile_weight=float(lightfm_like_cfg.get("profile_weight", 0.75)),
                        content_weight=float(lightfm_like_cfg.get("content_weight", 1.0)),
                        culture_weight=float(lightfm_like_cfg.get("culture_weight", 0.25)),
                        source_weight=float(lightfm_like_cfg.get("source_weight", 0.15)),
                        seed=int(lightfm_like_cfg.get("seed", 42)),
                        prefer_cuda=bool(lightfm_like_cfg.get("prefer_cuda", prefer_cuda)),
                    )
                device = torch.device("cuda" if prefer_cuda and torch.cuda.is_available() else "cpu")
                ranker, user_to_id, culture_to_id, source_to_id = load_content_bpr_mf(ckpt, map_location=str(device))
                recommend_fn = lambda user_id, target_culture, top_k, _tracks=tracks, _ints=interactions, _ranker=ranker, _users=user_to_id, _cultures=culture_to_id, _sources=source_to_id, _device=device: recommend_content_bpr_mf(
                    _ranker,
                    _users,
                    _cultures,
                    _sources,
                    _tracks,
                    _ints,
                    user_id,
                    target_culture,
                    k=top_k,
                    mf_weight=float(lightfm_like_cfg.get("mf_weight", 0.6)),
                    content_weight=float(lightfm_like_cfg.get("content_weight_rerank", 0.2)),
                    novelty_weight=float(lightfm_like_cfg.get("novelty_weight", 0.1)),
                    minority_weight=float(lightfm_like_cfg.get("minority_weight", 0.05)),
                    source_weight=float(lightfm_like_cfg.get("source_weight_rerank", 0.05)),
                    device=_device,
                )
            elif kind == "bpr_two_stage_hybrid":
                bpr_ckpt = str(bpr_cfg.get("checkpoint", out_dir / "bpr_mf.pt"))
                if not Path(bpr_ckpt).exists():
                    if not bool(bpr_cfg.get("train_if_missing", True)):
                        raise RuntimeError(f"BPR checkpoint missing and train_if_missing=false: {bpr_ckpt}")
                    train_bpr_mf(
                        tracks=tracks,
                        interactions=interactions,
                        out_path=bpr_ckpt,
                        latent_dim=int(bpr_cfg.get("latent_dim", 64)),
                        epochs=int(bpr_cfg.get("epochs", 10)),
                        batch_size=int(bpr_cfg.get("batch_size", 512)),
                        lr=float(bpr_cfg.get("lr", 5e-3)),
                        reg=float(bpr_cfg.get("reg", 1e-4)),
                        seed=int(bpr_cfg.get("seed", 42)),
                        prefer_cuda=bool(bpr_cfg.get("prefer_cuda", prefer_cuda)),
                    )
                ckpt = str(bpr_hybrid_cfg.get("checkpoint", out_dir / "bpr_two_stage_hybrid.pt"))
                if not Path(ckpt).exists():
                    if not bool(bpr_hybrid_cfg.get("train_if_missing", True)):
                        raise RuntimeError(f"BPR hybrid checkpoint missing and train_if_missing=false: {ckpt}")
                    train_bpr_two_stage_hybrid_ranker(
                        tracks=tracks,
                        interactions=interactions,
                        bpr_checkpoint=bpr_ckpt,
                        out_path=ckpt,
                        hidden_dim=int(bpr_hybrid_cfg.get("hidden_dim", 256)),
                        depth=int(bpr_hybrid_cfg.get("depth", 3)),
                        dropout=float(bpr_hybrid_cfg.get("dropout", 0.1)),
                        epochs=int(bpr_hybrid_cfg.get("epochs", 6)),
                        batch_size=int(bpr_hybrid_cfg.get("batch_size", 256)),
                        lr=float(bpr_hybrid_cfg.get("lr", 1e-3)),
                        recall_k=int(bpr_hybrid_cfg.get("recall_k", max(100, 5 * int(k)))),
                        negative_samples=int(bpr_hybrid_cfg.get("negative_samples", 6)),
                        hard_negative_ratio=float(bpr_hybrid_cfg.get("hard_negative_ratio", 0.8)),
                        seed=int(bpr_hybrid_cfg.get("seed", 42)),
                        prefer_cuda=bool(bpr_hybrid_cfg.get("prefer_cuda", prefer_cuda)),
                    )
                device = torch.device("cuda" if prefer_cuda and torch.cuda.is_available() else "cpu")
                bpr_model, user_to_id = load_bpr_mf(bpr_ckpt, map_location=str(device))
                ranker = load_bpr_two_stage_hybrid_ranker(ckpt, map_location=str(device))
                recommend_fn = lambda user_id, target_culture, top_k, _tracks=tracks, _ints=interactions, _ranker=ranker, _bpr=bpr_model, _users=user_to_id, _device=device, _cfg=bpr_hybrid_cfg: recommend_embedding_bpr_two_stage_hybrid(
                    _ranker,
                    _bpr,
                    _users,
                    _tracks,
                    _ints,
                    user_id,
                    target_culture,
                    k=top_k,
                    recall_k=int(_cfg.get("recall_k", max(100, 5 * int(top_k)))),
                    rerank_weight=float(_cfg.get("rerank_weight", 0.62)),
                    recall_weight=float(_cfg.get("recall_weight", 0.16)),
                    bpr_weight=float(_cfg.get("bpr_weight", 0.10)),
                    novelty_weight=float(_cfg.get("novelty_weight", 0.0)),
                    target_affinity_weight=float(_cfg.get("target_affinity_weight", 0.08)),
                    minority_weight=float(_cfg.get("minority_weight", 0.02)),
                    source_weight=float(_cfg.get("source_weight", 0.02)),
                    device=_device,
                )
            elif kind == "bpr_listwise_hybrid":
                bpr_ckpt = str(bpr_cfg.get("checkpoint", out_dir / "bpr_mf.pt"))
                if not Path(bpr_ckpt).exists():
                    if not bool(bpr_cfg.get("train_if_missing", True)):
                        raise RuntimeError(f"BPR checkpoint missing and train_if_missing=false: {bpr_ckpt}")
                    train_bpr_mf(
                        tracks=tracks,
                        interactions=interactions,
                        out_path=bpr_ckpt,
                        latent_dim=int(bpr_cfg.get("latent_dim", 64)),
                        epochs=int(bpr_cfg.get("epochs", 10)),
                        batch_size=int(bpr_cfg.get("batch_size", 512)),
                        lr=float(bpr_cfg.get("lr", 5e-3)),
                        reg=float(bpr_cfg.get("reg", 1e-4)),
                        seed=int(bpr_cfg.get("seed", 42)),
                        prefer_cuda=bool(bpr_cfg.get("prefer_cuda", prefer_cuda)),
                    )
                ckpt = str(bpr_listwise_hybrid_cfg.get("checkpoint", out_dir / "bpr_listwise_hybrid.pt"))
                if not Path(ckpt).exists():
                    if not bool(bpr_listwise_hybrid_cfg.get("train_if_missing", True)):
                        raise RuntimeError(f"BPR listwise hybrid checkpoint missing and train_if_missing=false: {ckpt}")
                    train_bpr_listwise_hybrid_ranker(
                        tracks=tracks,
                        interactions=interactions,
                        bpr_checkpoint=bpr_ckpt,
                        out_path=ckpt,
                        hidden_dim=int(bpr_listwise_hybrid_cfg.get("hidden_dim", 256)),
                        depth=int(bpr_listwise_hybrid_cfg.get("depth", 3)),
                        dropout=float(bpr_listwise_hybrid_cfg.get("dropout", 0.1)),
                        epochs=int(bpr_listwise_hybrid_cfg.get("epochs", 4)),
                        lr=float(bpr_listwise_hybrid_cfg.get("lr", 5e-4)),
                        recall_k=int(bpr_listwise_hybrid_cfg.get("recall_k", max(100, 5 * int(k)))),
                        warm_start_checkpoint=bpr_listwise_hybrid_cfg.get("warm_start_checkpoint"),
                        seed=int(bpr_listwise_hybrid_cfg.get("seed", 42)),
                        prefer_cuda=bool(bpr_listwise_hybrid_cfg.get("prefer_cuda", prefer_cuda)),
                    )
                device = torch.device("cuda" if prefer_cuda and torch.cuda.is_available() else "cpu")
                bpr_model, user_to_id = load_bpr_mf(bpr_ckpt, map_location=str(device))
                ranker = load_bpr_listwise_hybrid_ranker(ckpt, map_location=str(device))
                recommend_fn = lambda user_id, target_culture, top_k, _tracks=tracks, _ints=interactions, _ranker=ranker, _bpr=bpr_model, _users=user_to_id, _device=device, _cfg=bpr_listwise_hybrid_cfg: recommend_embedding_bpr_listwise_hybrid(
                    _ranker,
                    _bpr,
                    _users,
                    _tracks,
                    _ints,
                    user_id,
                    target_culture,
                    k=top_k,
                    recall_k=int(_cfg.get("recall_k", max(100, 5 * int(top_k)))),
                    rerank_weight=float(_cfg.get("rerank_weight", 0.62)),
                    recall_weight=float(_cfg.get("recall_weight", 0.14)),
                    bpr_weight=float(_cfg.get("bpr_weight", 0.10)),
                    novelty_weight=float(_cfg.get("novelty_weight", 0.02)),
                    target_affinity_weight=float(_cfg.get("target_affinity_weight", 0.06)),
                    minority_weight=float(_cfg.get("minority_weight", 0.04)),
                    source_weight=float(_cfg.get("source_weight", 0.02)),
                    device=_device,
                )
            else:
                raise ValueError(f"unsupported raw method kind: {kind}")
        elif family == "dcas":
            checkpoint = str(method_cfg["checkpoint"])
            device = torch.device("cuda" if prefer_cuda and torch.cuda.is_available() else "cpu")
            model, _ = load_checkpoint(checkpoint, map_location=str(device))
            if kind == "ot":
                recommend_fn = lambda user_id, target_culture, top_k, _model=model, _tracks=tracks, _ints=interactions, _device=device, _eps=method_cfg.get("epsilon", 0.1), _iters=method_cfg.get("iters", 200): recommend_ot(
                    model=_model,
                    tracks=_tracks,
                    interactions=_ints,
                    user_id=user_id,
                    target_culture=target_culture,
                    k=top_k,
                    device=_device,
                    epsilon=float(_eps),
                    iters=int(_iters),
                )[0]
            elif kind == "knn":
                recommend_fn = lambda user_id, target_culture, top_k, _model=model, _tracks=tracks, _ints=interactions, _device=device: recommend_knn(
                    model=_model,
                    tracks=_tracks,
                    interactions=_ints,
                    user_id=user_id,
                    target_culture=target_culture,
                    k=top_k,
                    device=_device,
                )[0]
            elif kind == "ot_calibrated":
                recommend_fn = lambda user_id, target_culture, top_k, _model=model, _tracks=tracks, _ints=interactions, _device=device, _cfg=method_cfg: recommend_ot_calibrated(
                    model=_model,
                    tracks=_tracks,
                    interactions=_ints,
                    user_id=user_id,
                    target_culture=target_culture,
                    k=top_k,
                    device=_device,
                    epsilon=float(_cfg.get("epsilon", 0.1)),
                    iters=int(_cfg.get("iters", 200)),
                    relevance_weight=float(_cfg.get("relevance_weight", 0.62)),
                    novelty_weight=float(_cfg.get("novelty_weight", 0.12)),
                    target_affinity_weight=float(_cfg.get("target_affinity_weight", 0.14)),
                    minority_weight=float(_cfg.get("minority_weight", 0.08)),
                    source_weight=float(_cfg.get("source_weight", 0.04)),
                    diversity_lambda=float(_cfg.get("diversity_lambda", 0.03)),
                )[0]
            elif kind == "knn_calibrated":
                recommend_fn = lambda user_id, target_culture, top_k, _model=model, _tracks=tracks, _ints=interactions, _device=device, _cfg=method_cfg: recommend_knn_calibrated(
                    model=_model,
                    tracks=_tracks,
                    interactions=_ints,
                    user_id=user_id,
                    target_culture=target_culture,
                    k=top_k,
                    device=_device,
                    relevance_weight=float(_cfg.get("relevance_weight", 0.62)),
                    novelty_weight=float(_cfg.get("novelty_weight", 0.12)),
                    target_affinity_weight=float(_cfg.get("target_affinity_weight", 0.14)),
                    minority_weight=float(_cfg.get("minority_weight", 0.08)),
                    source_weight=float(_cfg.get("source_weight", 0.04)),
                    diversity_lambda=float(_cfg.get("diversity_lambda", 0.03)),
                )[0]
            elif kind == "ot_open":
                recommend_fn = lambda user_id, target_culture, top_k, _model=model, _tracks=tracks, _ints=interactions, _device=device, _cfg=method_cfg: recommend_open_ot(
                    model=_model,
                    tracks=_tracks,
                    interactions=_ints,
                    user_id=user_id,
                    target_culture=target_culture,
                    k=top_k,
                    recall_k=int(_cfg.get("recall_k", max(50, 10 * int(top_k)))),
                    device=_device,
                    epsilon=float(_cfg.get("epsilon", 0.1)),
                    iters=int(_cfg.get("iters", 200)),
                    relevance_weight=float(_cfg.get("relevance_weight", 0.4)),
                    novelty_weight=float(_cfg.get("novelty_weight", 0.2)),
                    target_affinity_weight=float(_cfg.get("target_affinity_weight", 0.3)),
                    minority_weight=float(_cfg.get("minority_weight", 0.1)),
                    diversity_lambda=float(_cfg.get("diversity_lambda", 0.15)),
                )[0]
            elif kind == "knn_open":
                recommend_fn = lambda user_id, target_culture, top_k, _model=model, _tracks=tracks, _ints=interactions, _device=device, _cfg=method_cfg: recommend_open_knn(
                    model=_model,
                    tracks=_tracks,
                    interactions=_ints,
                    user_id=user_id,
                    target_culture=target_culture,
                    k=top_k,
                    recall_k=int(_cfg.get("recall_k", max(50, 10 * int(top_k)))),
                    device=_device,
                    relevance_weight=float(_cfg.get("relevance_weight", 0.4)),
                    novelty_weight=float(_cfg.get("novelty_weight", 0.2)),
                    target_affinity_weight=float(_cfg.get("target_affinity_weight", 0.3)),
                    minority_weight=float(_cfg.get("minority_weight", 0.1)),
                    diversity_lambda=float(_cfg.get("diversity_lambda", 0.15)),
                )[0]
            else:
                raise ValueError(f"unsupported dcas method kind: {kind}")
        else:
            raise ValueError(f"unsupported method family: {family}")

        eval_path = eval_dir / f"{name}.json"
        result = evaluate_callable_recommender(
            name=name,
            tracks=tracks,
            interactions=interactions,
            recommend_fn=recommend_fn,
            out_json=str(eval_path),
            k=int(k),
            bootstrap_samples=int(bootstrap_samples),
            bootstrap_seed=int(bootstrap_seed),
            minority_quantile=float(minority_quantile),
        )
        results_by_method[name] = result
        eval_paths[name] = str(eval_path)

    reference_method = str(cfg.get("reference_method", ""))
    comparisons: dict[str, Any] = {}
    if reference_method and reference_method in eval_paths:
        for name, path in eval_paths.items():
            if name == reference_method:
                continue
            cmp_json = cmp_dir / f"{name}_vs_{reference_method}.json"
            cmp_md = cmp_dir / f"{name}_vs_{reference_method}.md"
            comparisons[name] = compare_recommender_runs(
                base_eval_path=path,
                candidate_eval_path=eval_paths[reference_method],
                metrics=["serendipity", "cultural_calibration_kl", "minority_exposure_at_k"],
                bootstrap_samples=int(bootstrap_samples),
                permutation_samples=int(cfg.get("permutation_samples", 2000)),
                seed=int(bootstrap_seed) + 101,
                out_json=str(cmp_json),
                out_md=str(cmp_md),
            )

    summary: dict[str, Any] = {
        "suite_name": suite_name,
        "tracks": tracks_path,
        "interactions": interactions_path,
        "methods": {
            name: {
                "serendipity_mean": float(result["summary"]["serendipity_mean"]),
                "cultural_calibration_kl_mean": float(result["summary"]["cultural_calibration_kl_mean"]),
                "minority_exposure_at_k_mean": float(result["summary"]["minority_exposure_at_k_mean"]),
                "target_culture_prob_mean": float(result["summary"]["target_culture_prob_mean"]),
            }
            for name, result in results_by_method.items()
        },
        "reference_method": reference_method if reference_method else None,
        "comparisons_vs_reference": {
            name: {
                metric: {
                    "delta_mean": float(obj["metrics"][metric]["delta_mean"]),
                    "p_value_two_sided": float(obj["metrics"][metric]["p_value_two_sided"]),
                }
                for metric in obj["metrics"].keys()
            }
            for name, obj in comparisons.items()
        },
    }

    out_json = out_dir / "benchmark_summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    ordered_names = list(results_by_method.keys())
    lines = [
        f"# Recommender Benchmark: {suite_name}",
        "",
        f"- tracks: `{tracks_path}`",
        f"- interactions: `{interactions_path}`",
        f"- reference_method: `{reference_method}`" if reference_method else "- reference_method: `none`",
        "",
        "| method | serendipity | calibration_kl | minority@k | target_prob |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ordered_names:
        s = results_by_method[name]["summary"]
        lines.append(
            f"| {name} | {float(s['serendipity_mean']):.6f} | {float(s['cultural_calibration_kl_mean']):.6f} | "
            f"{float(s['minority_exposure_at_k_mean']):.6f} | {float(s['target_culture_prob_mean']):.6f} |"
        )
    if comparisons:
        lines.extend(
            [
                "",
                "## Reference Comparisons",
                "",
                "| baseline | metric | delta(reference-baseline) | p_value |",
                "|---|---|---:|---:|",
            ]
        )
        for name, obj in comparisons.items():
            for metric in ["serendipity", "cultural_calibration_kl", "minority_exposure_at_k"]:
                if metric not in obj["metrics"]:
                    continue
                lines.append(
                    f"| {name} | {metric} | {float(obj['metrics'][metric]['delta_mean']):+.6f} | {float(obj['metrics'][metric]['p_value_two_sided']):.6f} |"
                )
    (out_dir / "benchmark_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Run unified recommender benchmark suite.")
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    out = run_benchmark_suite(args.config)
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
