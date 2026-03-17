from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from dcas.data.interactions import Interaction
from dcas.data.npz_tracks import Tracks
from dcas.recommender import Recommendation


def _safe_softmax(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x.astype(np.float32)
    z = x - float(np.max(x))
    e = np.exp(z)
    return (e / max(1e-12, float(e.sum()))).astype(np.float32)


def _minmax(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x.astype(np.float32)
    lo = float(np.min(x))
    hi = float(np.max(x))
    if hi - lo <= 1e-12:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - lo) / (hi - lo)).astype(np.float32)


def _track_id_to_idx(tracks: Tracks) -> dict[str, int]:
    return {str(tid): int(i) for i, tid in enumerate(tracks.track_id.tolist())}


def _prepare_user_and_candidates(
    tracks: Tracks,
    interactions: list[Interaction],
    user_id: str,
    target_culture: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    track_id_to_idx = _track_id_to_idx(tracks)
    user_hist = [it for it in interactions if str(it.user_id) == str(user_id) and str(it.track_id) in track_id_to_idx]
    if not user_hist:
        raise ValueError(f"no interactions for user_id={user_id}")

    hist_idx = np.array([track_id_to_idx[str(it.track_id)] for it in user_hist], dtype=np.int64)
    hist_w = np.array([float(it.weight) for it in user_hist], dtype=np.float32)
    hist_w = hist_w / max(1e-12, float(hist_w.sum()))

    cand_idx = tracks.indices_of_cultures([target_culture]).astype(np.int64)
    if cand_idx.size == 0:
        raise ValueError(f"no tracks for target_culture={target_culture}")
    return hist_idx, hist_w, cand_idx


def _user_profile(emb: np.ndarray, hist_idx: np.ndarray, hist_w: np.ndarray) -> np.ndarray:
    return np.average(emb[hist_idx], axis=0, weights=hist_w).astype(np.float32)


def _popularity_by_track(interactions: list[Interaction]) -> dict[str, float]:
    out: dict[str, float] = {}
    for it in interactions:
        tid = str(it.track_id)
        out[tid] = float(out.get(tid, 0.0) + float(it.weight))
    return out


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b, axis=1)
    denom = np.maximum(1e-12, a_norm * b_norm)
    return (b @ a) / denom


def _finalize_embedding_recommendations(
    tracks: Tracks,
    hist_idx: np.ndarray,
    cand_idx: np.ndarray,
    cand_scores: np.ndarray,
    k: int,
) -> list[Recommendation]:
    hist_emb = tracks.embedding[hist_idx]
    cand_emb = tracks.embedding[cand_idx]
    top_local = np.argsort(-cand_scores)[: int(k)]
    recs: list[Recommendation] = []
    for j in top_local.tolist():
        idx = int(cand_idx[j])
        tid = str(tracks.track_id[idx])
        culture = str(tracks.culture[idx])
        score = float(cand_scores[j])
        d = np.linalg.norm(hist_emb - cand_emb[j], axis=1)
        rel = float(_safe_softmax(-d).max()) if d.size else 0.0
        unexp = float(np.mean(d)) if d.size else 0.0
        recs.append(
            Recommendation(
                track_id=tid,
                culture=culture,
                score=score,
                relevance=rel,
                unexpectedness=unexp,
            )
        )
    return recs


def recommend_popularity(
    tracks: Tracks,
    interactions: list[Interaction],
    user_id: str,
    target_culture: str,
    k: int = 20,
) -> list[Recommendation]:
    hist_idx, _, cand_idx = _prepare_user_and_candidates(tracks, interactions, user_id, target_culture)
    popularity = _popularity_by_track(interactions)
    scores = np.array([float(popularity.get(str(tracks.track_id[i]), 0.0)) for i in cand_idx.tolist()], dtype=np.float32)
    scores = _minmax(scores)
    return _finalize_embedding_recommendations(tracks, hist_idx, cand_idx, scores, k=k)


def recommend_embedding_cosine(
    tracks: Tracks,
    interactions: list[Interaction],
    user_id: str,
    target_culture: str,
    k: int = 20,
) -> list[Recommendation]:
    hist_idx, hist_w, cand_idx = _prepare_user_and_candidates(tracks, interactions, user_id, target_culture)
    user_vec = _user_profile(tracks.embedding, hist_idx, hist_w)
    scores = _cosine_similarity(user_vec, tracks.embedding[cand_idx]).astype(np.float32)
    return _finalize_embedding_recommendations(tracks, hist_idx, cand_idx, scores, k=k)


def recommend_embedding_knn(
    tracks: Tracks,
    interactions: list[Interaction],
    user_id: str,
    target_culture: str,
    k: int = 20,
) -> list[Recommendation]:
    hist_idx, hist_w, cand_idx = _prepare_user_and_candidates(tracks, interactions, user_id, target_culture)
    hist_emb = tracks.embedding[hist_idx]
    cand_emb = tracks.embedding[cand_idx]
    dist = np.linalg.norm(hist_emb[:, None, :] - cand_emb[None, :, :], axis=2)
    avg_dist = (hist_w[:, None] * dist).sum(axis=0)
    scores = _safe_softmax(-avg_dist)
    return _finalize_embedding_recommendations(tracks, hist_idx, cand_idx, scores, k=k)


def recommend_embedding_hybrid(
    tracks: Tracks,
    interactions: list[Interaction],
    user_id: str,
    target_culture: str,
    k: int = 20,
    cosine_weight: float = 0.4,
    knn_weight: float = 0.25,
    popularity_weight: float = 0.2,
    novelty_weight: float = 0.15,
) -> list[Recommendation]:
    hist_idx, hist_w, cand_idx = _prepare_user_and_candidates(tracks, interactions, user_id, target_culture)
    hist_emb = tracks.embedding[hist_idx]
    cand_emb = tracks.embedding[cand_idx]
    user_vec = _user_profile(tracks.embedding, hist_idx, hist_w)
    popularity = _popularity_by_track(interactions)

    cosine = _minmax(_cosine_similarity(user_vec, cand_emb))
    dist = np.linalg.norm(hist_emb[:, None, :] - cand_emb[None, :, :], axis=2)
    knn = _minmax(-(hist_w[:, None] * dist).sum(axis=0))
    novelty = _minmax(dist.mean(axis=0))
    pop = _minmax(
        np.array([float(popularity.get(str(tracks.track_id[i]), 0.0)) for i in cand_idx.tolist()], dtype=np.float32)
    )
    scores = (
        float(cosine_weight) * cosine
        + float(knn_weight) * knn
        + float(popularity_weight) * pop
        + float(novelty_weight) * novelty
    ).astype(np.float32)
    return _finalize_embedding_recommendations(tracks, hist_idx, cand_idx, scores, k=k)


def _pair_features(user_vec: np.ndarray, item_vec: np.ndarray) -> np.ndarray:
    return np.concatenate([user_vec, item_vec, np.abs(user_vec - item_vec), user_vec * item_vec], axis=0).astype(np.float32)


@dataclass(frozen=True)
class ShallowRankerConfig:
    input_dim: int
    hidden_dim: int = 128


class ShallowRanker(nn.Module):
    def __init__(self, cfg: ShallowRankerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.net = nn.Sequential(
            nn.Linear(int(cfg.input_dim), int(cfg.hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(cfg.hidden_dim), 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _make_training_examples(
    tracks: Tracks,
    interactions: list[Interaction],
    negative_samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    track_id_to_idx = _track_id_to_idx(tracks)
    all_idx = np.arange(len(tracks), dtype=np.int64)
    by_user: dict[str, list[Interaction]] = {}
    for it in interactions:
        by_user.setdefault(str(it.user_id), []).append(it)

    features: list[np.ndarray] = []
    labels: list[float] = []
    for user_id in sorted(by_user.keys()):
        rows = [it for it in by_user[user_id] if str(it.track_id) in track_id_to_idx]
        if len(rows) <= 1:
            continue
        pos_idx = np.array([track_id_to_idx[str(it.track_id)] for it in rows], dtype=np.int64)
        weights = np.array([float(it.weight) for it in rows], dtype=np.float32)
        weights = weights / max(1e-12, float(weights.sum()))
        hist_set = set(pos_idx.tolist())
        neg_pool = np.array([i for i in all_idx.tolist() if i not in hist_set], dtype=np.int64)
        if neg_pool.size == 0:
            continue
        for j, pos in enumerate(pos_idx.tolist()):
            mask = np.ones_like(pos_idx, dtype=bool)
            mask[j] = False
            context_idx = pos_idx[mask]
            context_w = weights[mask]
            if context_idx.size == 0:
                context_idx = pos_idx
                context_w = weights
            context_w = context_w / max(1e-12, float(context_w.sum()))
            user_vec = _user_profile(tracks.embedding, context_idx, context_w)
            features.append(_pair_features(user_vec, tracks.embedding[int(pos)]))
            labels.append(1.0)

            n_neg = min(int(negative_samples), int(neg_pool.size))
            neg_idx = rng.choice(neg_pool, size=n_neg, replace=False)
            for neg in neg_idx.tolist():
                features.append(_pair_features(user_vec, tracks.embedding[int(neg)]))
                labels.append(0.0)

    if not features:
        raise RuntimeError("could not build shallow ranker training examples")
    x = np.stack(features, axis=0).astype(np.float32)
    y = np.array(labels, dtype=np.float32)
    return x, y


def train_shallow_ranker(
    tracks: Tracks,
    interactions: list[Interaction],
    out_path: str | Path,
    hidden_dim: int = 128,
    epochs: int = 5,
    batch_size: int = 256,
    lr: float = 1e-3,
    negative_samples: int = 2,
    seed: int = 42,
    prefer_cuda: bool = False,
) -> dict[str, object]:
    x, y = _make_training_examples(
        tracks=tracks,
        interactions=interactions,
        negative_samples=int(negative_samples),
        seed=int(seed),
    )
    device = torch.device("cuda" if prefer_cuda and torch.cuda.is_available() else "cpu")
    ds = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    dl = DataLoader(ds, batch_size=min(int(batch_size), len(ds)), shuffle=True, drop_last=False)

    cfg = ShallowRankerConfig(input_dim=int(x.shape[1]), hidden_dim=int(hidden_dim))
    model = ShallowRanker(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr))
    loss_fn = nn.BCEWithLogitsLoss()
    history: list[dict[str, float]] = []
    for epoch in range(int(epochs)):
        model.train()
        losses: list[float] = []
        for batch_x, batch_y in dl:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            logits = model(batch_x)
            loss = loss_fn(logits, batch_y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu().item()))
        history.append({"epoch": float(epoch), "loss": float(np.mean(losses)) if losses else float("nan")})

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"cfg": asdict(cfg), "state_dict": model.state_dict(), "history": history}, str(out))
    return {"checkpoint": str(out), "history": history, "n_examples": int(x.shape[0])}


def load_shallow_ranker(path: str | Path, map_location: str | None = None) -> ShallowRanker:
    obj = torch.load(str(path), map_location=map_location)
    cfg = ShallowRankerConfig(**obj["cfg"])
    model = ShallowRanker(cfg)
    model.load_state_dict(obj["state_dict"])
    return model


def recommend_embedding_mlp(
    model: ShallowRanker,
    tracks: Tracks,
    interactions: list[Interaction],
    user_id: str,
    target_culture: str,
    k: int = 20,
    device: torch.device | None = None,
) -> list[Recommendation]:
    if device is None:
        device = torch.device("cpu")
    hist_idx, hist_w, cand_idx = _prepare_user_and_candidates(tracks, interactions, user_id, target_culture)
    user_vec = _user_profile(tracks.embedding, hist_idx, hist_w)
    feat = np.stack([_pair_features(user_vec, tracks.embedding[int(i)]) for i in cand_idx.tolist()], axis=0)
    model.eval()
    model.to(device)
    with torch.no_grad():
        logits = model(torch.from_numpy(feat).to(device)).detach().cpu().numpy().astype(np.float32)
    scores = _safe_softmax(logits)
    return _finalize_embedding_recommendations(tracks, hist_idx, cand_idx, scores, k=k)
