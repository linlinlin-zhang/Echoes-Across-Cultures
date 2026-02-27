from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from dcas.data.interactions import Interaction
from dcas.data.npz_tracks import Tracks
from dcas.models.dcas_vae import DCASModel
from dcas.ot.sinkhorn import sinkhorn_plan, squared_euclidean_cost


@dataclass(frozen=True)
class Recommendation:
    track_id: str
    culture: str
    score: float
    relevance: float
    unexpectedness: float


def _safe_kl(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = p.astype(np.float64)
    q = q.astype(np.float64)
    p = p / max(eps, p.sum())
    q = q / max(eps, q.sum())
    return float(np.sum(p * (np.log(p + eps) - np.log(q + eps))))


def _culture_centroids(
    zs_all: torch.Tensor,
    culture_all: np.ndarray,
    culture_names: list[str],
) -> torch.Tensor:
    centroids: list[torch.Tensor] = []
    for c in culture_names:
        idx = np.nonzero(culture_all == c)[0]
        if idx.size == 0:
            continue
        zc = zs_all[torch.from_numpy(idx).long()]
        centroids.append(zc.mean(dim=0, keepdim=True))
    if not centroids:
        raise ValueError("cannot build culture centroids: empty cultures")
    return torch.cat(centroids, dim=0)


def _soft_culture_distribution(
    zs_points: torch.Tensor,
    centroids: torch.Tensor,
    temperature: float = 1.0,
) -> np.ndarray:
    n_c = int(centroids.shape[0])
    if int(zs_points.shape[0]) == 0:
        return np.full((n_c,), 1.0 / max(1, n_c), dtype=np.float64)
    d = torch.cdist(zs_points, centroids)
    p = torch.softmax(-d / max(1e-6, float(temperature)), dim=1)
    return p.mean(dim=0).detach().cpu().numpy().astype(np.float64)


def _prepare_user_and_candidates(
    tracks: Tracks,
    interactions: list[Interaction],
    user_id: str,
    target_culture: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    track_id_to_idx = {str(tid): i for i, tid in enumerate(tracks.track_id.tolist())}
    user_hist = [it for it in interactions if it.user_id == user_id and it.track_id in track_id_to_idx]
    if not user_hist:
        raise ValueError(f"no interactions for user_id={user_id}")

    hist_idx = np.array([track_id_to_idx[it.track_id] for it in user_hist], dtype=np.int64)
    hist_w = np.array([float(it.weight) for it in user_hist], dtype=np.float64)
    hist_w = hist_w / max(1e-12, hist_w.sum())

    cand_idx = tracks.indices_of_cultures([target_culture])
    if cand_idx.size == 0:
        raise ValueError(f"no tracks for target_culture={target_culture}")
    return hist_idx, hist_w, cand_idx


def _finalize_recommendations(
    tracks: Tracks,
    cand_idx: np.ndarray,
    cand_scores: np.ndarray,
    za_hist: torch.Tensor,
    zs_hist: torch.Tensor,
    za_cand: torch.Tensor,
    zs_cand: torch.Tensor,
    zs_all: torch.Tensor,
    target_culture: str,
    k: int,
) -> tuple[list[Recommendation], dict[str, float]]:
    top_local = np.argsort(-cand_scores)[: int(k)]

    za_hist_cpu = za_hist.detach().cpu()
    zs_hist_cpu = zs_hist.detach().cpu()
    za_cand_cpu = za_cand.detach().cpu()
    zs_cand_cpu = zs_cand.detach().cpu()

    recs: list[Recommendation] = []
    for j in top_local.tolist():
        idx = int(cand_idx[j])
        tid = str(tracks.track_id[idx])
        cul = str(tracks.culture[idx])
        score = float(cand_scores[j])

        d_za = torch.cdist(za_cand_cpu[j : j + 1], za_hist_cpu).squeeze(0)
        relevance = float((-d_za).softmax(dim=0).max().item())

        d_zs = torch.cdist(zs_cand_cpu[j : j + 1], zs_hist_cpu).squeeze(0)
        unexpectedness = float(d_zs.mean().item())

        recs.append(Recommendation(track_id=tid, culture=cul, score=score, relevance=relevance, unexpectedness=unexpectedness))

    unexpected = np.array([r.unexpectedness for r in recs], dtype=np.float64)
    relevant = np.array([r.relevance for r in recs], dtype=np.float64)
    unexpected = unexpected / max(1e-12, float(unexpected.max()))
    relevant = relevant / max(1e-12, float(relevant.max()))
    serendipity = float(np.mean(unexpected * relevant))

    all_cultures = tracks.cultures()
    culture_arr = tracks.culture.astype(str)

    centroids = _culture_centroids(
        zs_all=zs_all.detach().cpu(),
        culture_all=culture_arr,
        culture_names=all_cultures,
    )
    rec_zs = zs_cand_cpu[torch.from_numpy(top_local).long()]
    rec_soft_dist = _soft_culture_distribution(rec_zs, centroids, temperature=1.0)
    hist_soft_dist = _soft_culture_distribution(zs_hist_cpu, centroids, temperature=1.0)

    target_dist = np.full((len(all_cultures),), 0.0, dtype=np.float64)
    if str(target_culture) in all_cultures:
        # Smooth the target prior to avoid infinite/near-infinite KL while
        # preserving "target culture should dominate" semantics.
        smoothing = 0.05
        n_c = max(1, len(all_cultures))
        target_idx = all_cultures.index(str(target_culture))
        off = smoothing / float(max(1, n_c - 1))
        target_dist[:] = off
        target_dist[target_idx] = 1.0 - smoothing
    else:
        target_dist[:] = 1.0 / max(1, len(all_cultures))
    target_prob = float(rec_soft_dist[all_cultures.index(str(target_culture))]) if str(target_culture) in all_cultures else float("nan")

    # Legacy calibration kept for backward compatibility of older reports.
    cultures = [r.culture for r in recs]
    rec_dist_hard = np.array([cultures.count(c) for c in all_cultures], dtype=np.float64)
    pool_dist = np.array([(tracks.culture == c).sum() for c in all_cultures], dtype=np.float64)
    calibration_kl_legacy = _safe_kl(rec_dist_hard, pool_dist)

    # New calibration: how close recommendation style-distribution is to target culture.
    calibration_kl = _safe_kl(rec_soft_dist, target_dist)
    user_alignment_kl = _safe_kl(rec_soft_dist, hist_soft_dist)

    metrics = {
        "serendipity": serendipity,
        "cultural_calibration_kl": calibration_kl,
        "cultural_calibration_kl_legacy": calibration_kl_legacy,
        "target_culture_prob_mean": target_prob,
        "user_culture_alignment_kl": user_alignment_kl,
    }
    return recs, metrics


def recommend_ot(
    model: DCASModel,
    tracks: Tracks,
    interactions: list[Interaction],
    user_id: str,
    target_culture: str,
    k: int = 20,
    device: torch.device | None = None,
    epsilon: float = 0.1,
    iters: int = 200,
) -> tuple[list[Recommendation], dict[str, float]]:
    if device is None:
        device = torch.device("cpu")
    model.eval()
    model.to(device)

    hist_idx, hist_w, cand_idx = _prepare_user_and_candidates(
        tracks=tracks,
        interactions=interactions,
        user_id=user_id,
        target_culture=target_culture,
    )

    x_all = torch.from_numpy(tracks.embedding).to(device)
    with torch.no_grad():
        _, zs_mu_all, za_mu_all = model.encode(x_all)

    za_hist = za_mu_all[torch.from_numpy(hist_idx).to(device)]
    zs_hist = zs_mu_all[torch.from_numpy(hist_idx).to(device)]
    za_cand = za_mu_all[torch.from_numpy(cand_idx).to(device)]
    zs_cand = zs_mu_all[torch.from_numpy(cand_idx).to(device)]

    a = torch.from_numpy(hist_w.astype(np.float32)).to(device)
    b = torch.full((cand_idx.shape[0],), 1.0 / cand_idx.shape[0], device=device)
    cost = squared_euclidean_cost(za_hist, za_cand)
    plan = sinkhorn_plan(a=a, b=b, cost=cost, epsilon=epsilon, iters=iters)

    # In balanced OT with fixed target marginal b, column mass is near-constant.
    # Rank candidates by OT-conditioned transport cost instead of marginal mass.
    col_mass = plan.sum(dim=0).clamp_min(1e-12)
    col_avg_cost = (plan * cost).sum(dim=0) / col_mass
    cand_scores = torch.softmax(-col_avg_cost, dim=0).detach().cpu().numpy()

    return _finalize_recommendations(
        tracks=tracks,
        cand_idx=cand_idx,
        cand_scores=cand_scores,
        za_hist=za_hist,
        zs_hist=zs_hist,
        za_cand=za_cand,
        zs_cand=zs_cand,
        zs_all=zs_mu_all,
        target_culture=str(target_culture),
        k=int(k),
    )


def recommend_knn(
    model: DCASModel,
    tracks: Tracks,
    interactions: list[Interaction],
    user_id: str,
    target_culture: str,
    k: int = 20,
    device: torch.device | None = None,
) -> tuple[list[Recommendation], dict[str, float]]:
    if device is None:
        device = torch.device("cpu")
    model.eval()
    model.to(device)

    hist_idx, hist_w, cand_idx = _prepare_user_and_candidates(
        tracks=tracks,
        interactions=interactions,
        user_id=user_id,
        target_culture=target_culture,
    )

    x_all = torch.from_numpy(tracks.embedding).to(device)
    with torch.no_grad():
        _, zs_mu_all, za_mu_all = model.encode(x_all)

    za_hist = za_mu_all[torch.from_numpy(hist_idx).to(device)]
    zs_hist = zs_mu_all[torch.from_numpy(hist_idx).to(device)]
    za_cand = za_mu_all[torch.from_numpy(cand_idx).to(device)]
    zs_cand = zs_mu_all[torch.from_numpy(cand_idx).to(device)]

    hist_w_t = torch.from_numpy(hist_w.astype(np.float32)).to(device)
    dist = torch.cdist(za_hist, za_cand)
    avg_dist = (hist_w_t.unsqueeze(1) * dist).sum(dim=0)
    cand_scores = torch.softmax(-avg_dist, dim=0).detach().cpu().numpy()

    return _finalize_recommendations(
        tracks=tracks,
        cand_idx=cand_idx,
        cand_scores=cand_scores,
        za_hist=za_hist,
        zs_hist=zs_hist,
        za_cand=za_cand,
        zs_cand=zs_cand,
        zs_all=zs_mu_all,
        target_culture=str(target_culture),
        k=int(k),
    )
