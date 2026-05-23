from __future__ import annotations

import numpy as np
import torch

from dcas.data.npz_tracks import Tracks
from dcas.models.dcas_vae import DCASModel
from dcas.models.losses import entropy_from_logits


def _normalized(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    lo = float(np.min(values))
    hi = float(np.max(values))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-8:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - lo) / (hi - lo)).astype(np.float32)


def _culture_centroid_entropy(
    za_mu: torch.Tensor,
    cultures: np.ndarray,
    temperature: float = 1.0,
) -> np.ndarray:
    culture_names = sorted({str(c) for c in cultures.tolist()})
    centroids: list[torch.Tensor] = []
    for culture in culture_names:
        mask = torch.tensor(cultures == culture, device=za_mu.device)
        centroid = za_mu[mask].mean(dim=0, keepdim=True)
        centroids.append(centroid)
    centroid_tensor = torch.cat(centroids, dim=0)
    dists = torch.cdist(za_mu, centroid_tensor)
    logits = -dists / max(float(temperature), 1e-6)
    probs = torch.softmax(logits, dim=-1)
    ent = -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=-1)
    return ent.detach().cpu().numpy().astype(np.float32)


def _culture_neighbor_entropy(
    za_mu: torch.Tensor,
    cultures: np.ndarray,
    k_neighbors: int = 12,
) -> np.ndarray:
    n = int(za_mu.shape[0])
    if n <= 1:
        return np.zeros((n,), dtype=np.float32)
    dists = torch.cdist(za_mu, za_mu)
    dists.fill_diagonal_(float("inf"))
    k = max(1, min(int(k_neighbors), n - 1))
    nn_idx = torch.topk(dists, k=k, largest=False).indices.detach().cpu().numpy()
    culture_names = sorted({str(c) for c in cultures.tolist()})
    culture_to_idx = {c: i for i, c in enumerate(culture_names)}
    entropies = np.zeros((n,), dtype=np.float32)
    for i in range(n):
        counts = np.zeros((len(culture_names),), dtype=np.float32)
        for j in nn_idx[i].tolist():
            counts[culture_to_idx[str(cultures[j])]] += 1.0
        probs = counts / max(float(counts.sum()), 1.0)
        probs = np.clip(probs, 1e-8, 1.0)
        entropies[i] = float(-(probs * np.log(probs)).sum())
    return entropies


def rank_by_uncertainty(
    model: DCASModel,
    tracks: Tracks,
    device: torch.device | None = None,
    batch_size: int = 512,
    method: str = "auto",
    culture_temperature: float = 1.0,
) -> list[tuple[str, float]]:
    if device is None:
        device = torch.device("cpu")
    model.eval()
    model.to(device)

    x = torch.from_numpy(tracks.embedding.astype(np.float32))
    if method not in {"auto", "affect_entropy", "culture_centroid_entropy", "hybrid"}:
        raise ValueError(f"unsupported uncertainty method: {method}")

    za_mu_parts: list[torch.Tensor] = []
    affect_scores: list[float] = []
    with torch.no_grad():
        for start in range(0, x.shape[0], int(batch_size)):
            xb = x[start : start + int(batch_size)].to(device)
            _, _, za_mu = model.encode(xb)
            za_mu_parts.append(za_mu.detach())
            if method in {"affect_entropy", "hybrid"} or (
                method == "auto" and float(model.cfg.lambda_affect) > 0 and tracks.affect_label is not None
            ):
                logits = model.affect_head(za_mu)
                ent = entropy_from_logits(logits).detach().cpu().numpy()
                affect_scores.extend(float(v) for v in ent.tolist())

    za_mu_all = torch.cat(za_mu_parts, dim=0)
    if method == "auto":
        if float(model.cfg.lambda_affect) > 0 and tracks.affect_label is not None:
            chosen = "hybrid"
        else:
            chosen = "culture_centroid_entropy"
    else:
        chosen = method

    centroid_scores = _culture_centroid_entropy(
        za_mu=za_mu_all,
        cultures=tracks.culture,
        temperature=float(culture_temperature),
    )
    neighbor_scores = _culture_neighbor_entropy(
        za_mu=za_mu_all,
        cultures=tracks.culture,
        k_neighbors=min(12, max(1, int(za_mu_all.shape[0]) - 1)),
    )
    if float(np.std(centroid_scores)) < 1e-6:
        culture_scores = neighbor_scores
    else:
        culture_scores = 0.5 * _normalized(centroid_scores) + 0.5 * _normalized(neighbor_scores)
    if chosen == "culture_centroid_entropy":
        final_scores = culture_scores
    elif chosen == "affect_entropy":
        final_scores = np.asarray(affect_scores, dtype=np.float32)
    else:
        affect_arr = np.asarray(affect_scores, dtype=np.float32)
        final_scores = 0.5 * _normalized(culture_scores) + 0.5 * _normalized(affect_arr)

    scores: list[tuple[str, float]] = []
    for tid, s in zip(tracks.track_id.tolist(), final_scores.tolist()):
        scores.append((str(tid), float(s)))
    scores.sort(key=lambda t: -t[1])
    return scores

