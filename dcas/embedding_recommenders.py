from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import pickle

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
    scores = np.array(
        [float(popularity.get(str(tracks.track_id[i]), 0.0)) for i in cand_idx.tolist()],
        dtype=np.float32,
    )
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
        np.array(
            [float(popularity.get(str(tracks.track_id[i]), 0.0)) for i in cand_idx.tolist()],
            dtype=np.float32,
        )
    )
    scores = (
        float(cosine_weight) * cosine
        + float(knn_weight) * knn
        + float(popularity_weight) * pop
        + float(novelty_weight) * novelty
    ).astype(np.float32)
    return _finalize_embedding_recommendations(tracks, hist_idx, cand_idx, scores, k=k)


def _pair_features(user_vec: np.ndarray, item_vec: np.ndarray) -> np.ndarray:
    return np.concatenate([user_vec, item_vec, np.abs(user_vec - item_vec), user_vec * item_vec], axis=0).astype(
        np.float32
    )


def _hybrid_model_feature_matrix(
    user_vec: np.ndarray,
    cand_emb: np.ndarray,
    scalar_features: np.ndarray,
) -> np.ndarray:
    user_block = np.repeat(user_vec.reshape(1, -1), cand_emb.shape[0], axis=0).astype(np.float32)
    diff = np.abs(user_block - cand_emb).astype(np.float32)
    prod = (user_block * cand_emb).astype(np.float32)
    return np.concatenate(
        [
            scalar_features.astype(np.float32),
            user_block,
            cand_emb.astype(np.float32),
            diff,
            prod,
        ],
        axis=1,
    ).astype(np.float32)


def _culture_centroid_embeddings(tracks: Tracks) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for culture in tracks.cultures():
        idx = tracks.indices_of_cultures([culture]).astype(np.int64)
        if idx.size <= 0:
            continue
        out[str(culture)] = tracks.embedding[idx].mean(axis=0).astype(np.float32)
    return out


def _source_inverse_frequency(tracks: Tracks) -> dict[str, float]:
    if tracks.source_dataset is None:
        return {}
    src = np.array([str(x) for x in tracks.source_dataset.tolist()], dtype=object)
    unique, counts = np.unique(src, return_counts=True)
    return {str(k): 1.0 / float(v) for k, v in zip(unique.tolist(), counts.tolist())}


def _candidate_feature_table(
    tracks: Tracks,
    interactions: list[Interaction],
    hist_idx: np.ndarray,
    hist_w: np.ndarray,
    cand_idx: np.ndarray,
    target_culture: str,
    pop_by_track: dict[str, float],
    source_inv_freq: dict[str, float],
    culture_centroids: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    hist_emb = tracks.embedding[hist_idx]
    cand_emb = tracks.embedding[cand_idx]
    user_vec = _user_profile(tracks.embedding, hist_idx, hist_w)

    cosine_raw = _cosine_similarity(user_vec, cand_emb)
    dist = np.linalg.norm(hist_emb[:, None, :] - cand_emb[None, :, :], axis=2)
    weighted_dist = (hist_w[:, None] * dist).sum(axis=0)
    knn_raw = -weighted_dist
    max_hist_raw = -dist.min(axis=0)
    mean_hist_raw = -dist.mean(axis=0)
    novelty_raw = dist.mean(axis=0)
    popularity_raw = np.array(
        [float(pop_by_track.get(str(tracks.track_id[i]), 0.0)) for i in cand_idx.tolist()],
        dtype=np.float32,
    )
    minority_raw = -popularity_raw

    centroid = culture_centroids.get(str(target_culture))
    if centroid is None:
        target_affinity_raw = np.zeros((cand_idx.shape[0],), dtype=np.float32)
    else:
        target_affinity_raw = _cosine_similarity(np.asarray(centroid, dtype=np.float32), cand_emb).astype(np.float32)

    if tracks.source_dataset is not None:
        hist_sources = [str(tracks.source_dataset[int(i)]) for i in hist_idx.tolist()]
        total_hist = max(1, len(hist_sources))
        hist_source_share: dict[str, float] = {}
        for src in hist_sources:
            hist_source_share[src] = float(hist_source_share.get(src, 0.0) + 1.0 / float(total_hist))
        cand_sources = [str(tracks.source_dataset[int(i)]) for i in cand_idx.tolist()]
        source_pref_raw = np.array(
            [float(hist_source_share.get(src, 0.0)) for src in cand_sources],
            dtype=np.float32,
        )
        source_inv_raw = np.array(
            [float(source_inv_freq.get(src, 0.0)) for src in cand_sources],
            dtype=np.float32,
        )
    else:
        source_pref_raw = np.zeros((cand_idx.shape[0],), dtype=np.float32)
        source_inv_raw = np.zeros((cand_idx.shape[0],), dtype=np.float32)

    cosine = _minmax(cosine_raw)
    knn = _minmax(knn_raw)
    max_hist = _minmax(max_hist_raw)
    mean_hist = _minmax(mean_hist_raw)
    novelty = _minmax(novelty_raw)
    popularity = _minmax(popularity_raw)
    minority = _minmax(minority_raw)
    target_affinity = _minmax(target_affinity_raw)
    source_pref = _minmax(source_pref_raw)
    source_inv = _minmax(source_inv_raw)

    feature_mat = np.stack(
        [
            cosine,
            knn,
            max_hist,
            mean_hist,
            novelty,
            popularity,
            minority,
            target_affinity,
            source_pref,
            source_inv,
        ],
        axis=1,
    ).astype(np.float32)

    recall_score = (
        0.24 * cosine
        + 0.20 * knn
        + 0.16 * max_hist
        + 0.10 * mean_hist
        + 0.14 * popularity
        + 0.10 * target_affinity
        + 0.06 * source_pref
    ).astype(np.float32)
    return feature_mat, recall_score


@dataclass(frozen=True)
class TwoStageHybridConfig:
    input_dim: int
    hidden_dim: int = 128
    depth: int = 3
    dropout: float = 0.1


class TwoStageHybridRanker(nn.Module):
    def __init__(self, cfg: TwoStageHybridConfig) -> None:
        super().__init__()
        self.cfg = cfg
        layers: list[nn.Module] = []
        in_dim = int(cfg.input_dim)
        hidden_dim = int(cfg.hidden_dim)
        depth = max(1, int(cfg.depth))
        for _ in range(depth - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            if float(cfg.dropout) > 0:
                layers.append(nn.Dropout(float(cfg.dropout)))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


@dataclass(frozen=True)
class BPRMFConfig:
    n_users: int
    n_items: int
    latent_dim: int = 64
    reg: float = 1e-4


class BPRMF(nn.Module):
    def __init__(self, cfg: BPRMFConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.user_factors = nn.Embedding(int(cfg.n_users), int(cfg.latent_dim))
        self.item_factors = nn.Embedding(int(cfg.n_items), int(cfg.latent_dim))
        self.item_bias = nn.Embedding(int(cfg.n_items), 1)
        nn.init.normal_(self.user_factors.weight, std=0.05)
        nn.init.normal_(self.item_factors.weight, std=0.05)
        nn.init.zeros_(self.item_bias.weight)

    def score(self, user_idx: torch.Tensor, item_idx: torch.Tensor) -> torch.Tensor:
        u = self.user_factors(user_idx)
        i = self.item_factors(item_idx)
        b = self.item_bias(item_idx).squeeze(-1)
        return (u * i).sum(dim=-1) + b

    def forward(
        self, user_idx: torch.Tensor, pos_idx: torch.Tensor, neg_idx: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pos_score = self.score(user_idx, pos_idx)
        neg_score = self.score(user_idx, neg_idx)
        return pos_score, neg_score


@dataclass(frozen=True)
class ContentBPRMFConfig:
    n_users: int
    n_items: int
    item_feat_dim: int
    n_cultures: int
    n_sources: int
    latent_dim: int = 64
    reg: float = 1e-4
    profile_weight: float = 0.75
    content_weight: float = 1.0
    culture_weight: float = 0.25
    source_weight: float = 0.15


class ContentBPRMF(nn.Module):
    def __init__(self, cfg: ContentBPRMFConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.user_factors = nn.Embedding(int(cfg.n_users), int(cfg.latent_dim))
        self.item_factors = nn.Embedding(int(cfg.n_items), int(cfg.latent_dim))
        self.item_bias = nn.Embedding(int(cfg.n_items), 1)
        self.user_proj = nn.Linear(int(cfg.item_feat_dim), int(cfg.latent_dim), bias=False)
        self.item_proj = nn.Linear(int(cfg.item_feat_dim), int(cfg.latent_dim), bias=False)
        self.culture_factors = nn.Embedding(max(1, int(cfg.n_cultures)), int(cfg.latent_dim))
        self.source_factors = nn.Embedding(max(1, int(cfg.n_sources)), int(cfg.latent_dim))
        nn.init.normal_(self.user_factors.weight, std=0.05)
        nn.init.normal_(self.item_factors.weight, std=0.05)
        nn.init.normal_(self.user_proj.weight, std=0.05)
        nn.init.normal_(self.item_proj.weight, std=0.05)
        nn.init.normal_(self.culture_factors.weight, std=0.05)
        nn.init.normal_(self.source_factors.weight, std=0.05)
        nn.init.zeros_(self.item_bias.weight)

    def _user_latent(self, user_idx: torch.Tensor, user_profile: torch.Tensor) -> torch.Tensor:
        return self.user_factors(user_idx) + float(self.cfg.profile_weight) * self.user_proj(user_profile)

    def _item_latent(
        self,
        item_idx: torch.Tensor,
        item_feat: torch.Tensor,
        item_culture_idx: torch.Tensor,
        item_source_idx: torch.Tensor,
    ) -> torch.Tensor:
        return (
            self.item_factors(item_idx)
            + float(self.cfg.content_weight) * self.item_proj(item_feat)
            + float(self.cfg.culture_weight) * self.culture_factors(item_culture_idx)
            + float(self.cfg.source_weight) * self.source_factors(item_source_idx)
        )

    def score_with_features(
        self,
        user_idx: torch.Tensor,
        item_idx: torch.Tensor,
        user_profile: torch.Tensor,
        item_features: torch.Tensor,
        item_culture_ids: torch.Tensor,
        item_source_ids: torch.Tensor,
    ) -> torch.Tensor:
        user_latent = self._user_latent(user_idx, user_profile)
        item_latent = self._item_latent(
            item_idx,
            item_features[item_idx],
            item_culture_ids[item_idx],
            item_source_ids[item_idx],
        )
        return (user_latent * item_latent).sum(dim=-1) + self.item_bias(item_idx).squeeze(-1)

    def forward(
        self,
        user_idx: torch.Tensor,
        pos_idx: torch.Tensor,
        neg_idx: torch.Tensor,
        user_profile: torch.Tensor,
        item_features: torch.Tensor,
        item_culture_ids: torch.Tensor,
        item_source_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pos_score = self.score_with_features(
            user_idx=user_idx,
            item_idx=pos_idx,
            user_profile=user_profile,
            item_features=item_features,
            item_culture_ids=item_culture_ids,
            item_source_ids=item_source_ids,
        )
        neg_score = self.score_with_features(
            user_idx=user_idx,
            item_idx=neg_idx,
            user_profile=user_profile,
            item_features=item_features,
            item_culture_ids=item_culture_ids,
            item_source_ids=item_source_ids,
        )
        return pos_score, neg_score


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
        history.append(
            {
                "epoch": float(epoch),
                "loss": float(np.mean(losses)) if losses else float("nan"),
            }
        )

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"cfg": asdict(cfg), "state_dict": model.state_dict(), "history": history},
        str(out),
    )
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
    feat = np.stack(
        [_pair_features(user_vec, tracks.embedding[int(i)]) for i in cand_idx.tolist()],
        axis=0,
    )
    model.eval()
    model.to(device)
    with torch.no_grad():
        logits = model(torch.from_numpy(feat).to(device)).detach().cpu().numpy().astype(np.float32)
    scores = _safe_softmax(logits)
    return _finalize_embedding_recommendations(tracks, hist_idx, cand_idx, scores, k=k)


def _make_two_stage_training_examples(
    tracks: Tracks,
    interactions: list[Interaction],
    negative_samples: int,
    recall_k: int,
    hard_negative_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    track_id_to_idx = _track_id_to_idx(tracks)
    by_user: dict[str, list[Interaction]] = {}
    for it in interactions:
        by_user.setdefault(str(it.user_id), []).append(it)

    pop_by_track = _popularity_by_track(interactions)
    source_inv_freq = _source_inverse_frequency(tracks)
    culture_centroids = _culture_centroid_embeddings(tracks)

    features: list[np.ndarray] = []
    labels: list[float] = []
    for user_id in sorted(by_user.keys()):
        rows = [it for it in by_user[user_id] if str(it.track_id) in track_id_to_idx]
        if len(rows) <= 1:
            continue
        pos_idx = np.array([track_id_to_idx[str(it.track_id)] for it in rows], dtype=np.int64)
        weights = np.array([float(it.weight) for it in rows], dtype=np.float32)
        weights = weights / max(1e-12, float(weights.sum()))
        seen = {int(i) for i in pos_idx.tolist()}

        for j, pos in enumerate(pos_idx.tolist()):
            mask = np.ones_like(pos_idx, dtype=bool)
            mask[int(j)] = False
            context_idx = pos_idx[mask]
            context_w = weights[mask]
            if context_idx.size <= 0:
                continue
            context_w = context_w / max(1e-12, float(context_w.sum()))
            user_vec = _user_profile(tracks.embedding, context_idx, context_w)
            target_culture = str(tracks.culture[int(pos)])
            cand_all = tracks.indices_of_cultures([target_culture]).astype(np.int64)
            cand_idx = np.array(
                [int(i) for i in cand_all.tolist() if int(i) == int(pos) or int(i) not in seen],
                dtype=np.int64,
            )
            if cand_idx.size <= 1:
                continue
            feature_mat, recall_score = _candidate_feature_table(
                tracks=tracks,
                interactions=interactions,
                hist_idx=context_idx,
                hist_w=context_w,
                cand_idx=cand_idx,
                target_culture=target_culture,
                pop_by_track=pop_by_track,
                source_inv_freq=source_inv_freq,
                culture_centroids=culture_centroids,
            )
            model_feature_mat = _hybrid_model_feature_matrix(
                user_vec=user_vec,
                cand_emb=tracks.embedding[cand_idx],
                scalar_features=feature_mat,
            )
            pos_local = np.nonzero(cand_idx == int(pos))[0]
            if pos_local.size <= 0:
                continue
            pos_local_idx = int(pos_local[0])
            features.append(model_feature_mat[pos_local_idx])
            labels.append(1.0)

            ordered = np.argsort(-recall_score)
            ordered = ordered[ordered != pos_local_idx]
            if ordered.size <= 0:
                continue

            hard_count = min(
                int(round(float(negative_samples) * float(hard_negative_ratio))),
                int(ordered.size),
            )
            easy_count = max(0, int(negative_samples) - int(hard_count))
            chosen: list[int] = []
            if hard_count > 0:
                hard_pool = ordered[: min(int(recall_k), int(ordered.size))]
                if hard_pool.size > 0:
                    take = min(int(hard_count), int(hard_pool.size))
                    chosen.extend(rng.choice(hard_pool, size=take, replace=False).tolist())
            if easy_count > 0:
                easy_pool = np.array(
                    [int(x) for x in ordered.tolist() if int(x) not in set(chosen)],
                    dtype=np.int64,
                )
                if easy_pool.size > 0:
                    take = min(int(easy_count), int(easy_pool.size))
                    chosen.extend(rng.choice(easy_pool, size=take, replace=False).tolist())
            if not chosen:
                chosen = ordered[: min(int(negative_samples), int(ordered.size))].tolist()

            for neg_local_idx in chosen:
                features.append(model_feature_mat[int(neg_local_idx)])
                labels.append(0.0)

    if not features:
        raise RuntimeError("could not build two-stage hybrid training examples")
    x = np.stack(features, axis=0).astype(np.float32)
    y = np.array(labels, dtype=np.float32)
    return x, y


def train_two_stage_hybrid_ranker(
    tracks: Tracks,
    interactions: list[Interaction],
    out_path: str | Path,
    hidden_dim: int = 128,
    depth: int = 3,
    dropout: float = 0.1,
    epochs: int = 6,
    batch_size: int = 256,
    lr: float = 1e-3,
    negative_samples: int = 4,
    recall_k: int = 80,
    hard_negative_ratio: float = 0.75,
    seed: int = 42,
    prefer_cuda: bool = False,
) -> dict[str, object]:
    x, y = _make_two_stage_training_examples(
        tracks=tracks,
        interactions=interactions,
        negative_samples=int(negative_samples),
        recall_k=int(recall_k),
        hard_negative_ratio=float(hard_negative_ratio),
        seed=int(seed),
    )
    device = torch.device("cuda" if prefer_cuda and torch.cuda.is_available() else "cpu")
    ds = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    dl = DataLoader(ds, batch_size=min(int(batch_size), len(ds)), shuffle=True, drop_last=False)

    cfg = TwoStageHybridConfig(
        input_dim=int(x.shape[1]),
        hidden_dim=int(hidden_dim),
        depth=int(depth),
        dropout=float(dropout),
    )
    model = TwoStageHybridRanker(cfg).to(device)
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
        history.append(
            {
                "epoch": float(epoch),
                "loss": float(np.mean(losses)) if losses else float("nan"),
            }
        )

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"cfg": asdict(cfg), "state_dict": model.state_dict(), "history": history},
        str(out),
    )
    return {"checkpoint": str(out), "history": history, "n_examples": int(x.shape[0])}


def load_two_stage_hybrid_ranker(path: str | Path, map_location: str | None = None) -> TwoStageHybridRanker:
    obj = torch.load(str(path), map_location=map_location)
    cfg = TwoStageHybridConfig(**obj["cfg"])
    model = TwoStageHybridRanker(cfg)
    model.load_state_dict(obj["state_dict"])
    return model


def recommend_embedding_two_stage_hybrid(
    model: TwoStageHybridRanker,
    tracks: Tracks,
    interactions: list[Interaction],
    user_id: str,
    target_culture: str,
    k: int = 20,
    recall_k: int = 80,
    blend_weight: float = 0.25,
    device: torch.device | None = None,
) -> list[Recommendation]:
    if device is None:
        device = torch.device("cpu")
    hist_idx, hist_w, cand_idx = _prepare_user_and_candidates(tracks, interactions, user_id, target_culture)
    user_vec = _user_profile(tracks.embedding, hist_idx, hist_w)
    pop_by_track = _popularity_by_track(interactions)
    source_inv_freq = _source_inverse_frequency(tracks)
    culture_centroids = _culture_centroid_embeddings(tracks)
    feature_mat, recall_score = _candidate_feature_table(
        tracks=tracks,
        interactions=interactions,
        hist_idx=hist_idx,
        hist_w=hist_w,
        cand_idx=cand_idx,
        target_culture=target_culture,
        pop_by_track=pop_by_track,
        source_inv_freq=source_inv_freq,
        culture_centroids=culture_centroids,
    )
    recall_n = min(max(int(k), int(recall_k)), int(cand_idx.shape[0]))
    recall_local = np.argsort(-recall_score)[:recall_n]
    cand_idx_recall = cand_idx[recall_local]
    feat_recall = _hybrid_model_feature_matrix(
        user_vec=user_vec,
        cand_emb=tracks.embedding[cand_idx_recall],
        scalar_features=feature_mat[recall_local],
    )

    model.eval()
    model.to(device)
    with torch.no_grad():
        logits = model(torch.from_numpy(feat_recall).to(device)).detach().cpu().numpy().astype(np.float32)
    rerank = _minmax(1.0 / (1.0 + np.exp(-logits)))
    recall_norm = _minmax(recall_score[recall_local])
    final_scores = ((1.0 - float(blend_weight)) * rerank + float(blend_weight) * recall_norm).astype(np.float32)
    return _finalize_embedding_recommendations(tracks, hist_idx, cand_idx_recall, final_scores, k=k)


def _build_bpr_training_state(
    tracks: Tracks,
    interactions: list[Interaction],
) -> tuple[dict[str, int], np.ndarray, np.ndarray, np.ndarray, dict[int, np.ndarray]]:
    track_id_to_idx = _track_id_to_idx(tracks)
    user_to_id = {str(u): i for i, u in enumerate(sorted({str(it.user_id) for it in interactions}))}
    user_seen: dict[int, set[int]] = {int(i): set() for i in user_to_id.values()}
    user_idx: list[int] = []
    pos_idx: list[int] = []
    weights: list[float] = []
    for it in interactions:
        tid = track_id_to_idx.get(str(it.track_id))
        uid = user_to_id.get(str(it.user_id))
        if tid is None or uid is None:
            continue
        user_idx.append(int(uid))
        pos_idx.append(int(tid))
        weights.append(float(it.weight))
        user_seen[int(uid)].add(int(tid))
    all_items = np.arange(len(tracks), dtype=np.int64)
    neg_pools = {
        int(uid): np.array([int(i) for i in all_items.tolist() if int(i) not in seen], dtype=np.int64)
        for uid, seen in user_seen.items()
    }
    return (
        user_to_id,
        np.array(user_idx, dtype=np.int64),
        np.array(pos_idx, dtype=np.int64),
        np.array(weights, dtype=np.float32),
        neg_pools,
    )


def _user_content_profiles(
    tracks: Tracks,
    interactions: list[Interaction],
    user_to_id: dict[str, int],
    track_id_to_idx: dict[str, int],
) -> np.ndarray:
    profiles = np.zeros((len(user_to_id), int(tracks.embedding.shape[1])), dtype=np.float32)
    totals = np.zeros((len(user_to_id),), dtype=np.float32)
    for it in interactions:
        tid = track_id_to_idx.get(str(it.track_id))
        uid = user_to_id.get(str(it.user_id))
        if tid is None or uid is None:
            continue
        w = max(1e-3, float(it.weight))
        profiles[int(uid)] += tracks.embedding[int(tid)].astype(np.float32) * float(w)
        totals[int(uid)] += float(w)
    mask = totals > 0
    profiles[mask] = profiles[mask] / np.maximum(totals[mask].reshape(-1, 1), 1e-12)
    return profiles.astype(np.float32)


def _item_feature_id_tables(
    tracks: Tracks,
) -> tuple[dict[str, int], dict[str, int], np.ndarray, np.ndarray]:
    culture_values = [str(x) for x in tracks.culture.tolist()]
    culture_to_id = {str(name): int(i) for i, name in enumerate(sorted(set(culture_values)))}
    item_culture_ids = np.array([int(culture_to_id[str(x)]) for x in culture_values], dtype=np.int64)

    if tracks.source_dataset is None:
        source_to_id = {"__none__": 0}
        item_source_ids = np.zeros((len(tracks),), dtype=np.int64)
    else:
        source_values = [str(x) for x in tracks.source_dataset.tolist()]
        source_to_id = {str(name): int(i) for i, name in enumerate(sorted(set(source_values)))}
        item_source_ids = np.array([int(source_to_id[str(x)]) for x in source_values], dtype=np.int64)
    return culture_to_id, source_to_id, item_culture_ids, item_source_ids


def train_bpr_mf(
    tracks: Tracks,
    interactions: list[Interaction],
    out_path: str | Path,
    latent_dim: int = 64,
    epochs: int = 10,
    batch_size: int = 512,
    lr: float = 5e-3,
    reg: float = 1e-4,
    seed: int = 42,
    prefer_cuda: bool = False,
) -> dict[str, object]:
    rng = np.random.default_rng(int(seed))
    user_to_id, user_idx, pos_idx, weights, neg_pools = _build_bpr_training_state(
        tracks=tracks, interactions=interactions
    )
    if user_idx.size <= 0:
        raise RuntimeError("could not build BPR training state")
    device = torch.device("cuda" if prefer_cuda and torch.cuda.is_available() else "cpu")
    cfg = BPRMFConfig(
        n_users=int(len(user_to_id)),
        n_items=int(len(tracks)),
        latent_dim=int(latent_dim),
        reg=float(reg),
    )
    model = BPRMF(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr))

    order = np.arange(user_idx.shape[0], dtype=np.int64)
    history: list[dict[str, float]] = []
    for epoch in range(int(epochs)):
        rng.shuffle(order)
        losses: list[float] = []
        model.train()
        for start in range(0, int(order.shape[0]), int(batch_size)):
            batch_ids = order[start : start + int(batch_size)]
            bu = user_idx[batch_ids]
            bp = pos_idx[batch_ids]
            bw = weights[batch_ids]
            bn = np.empty_like(bp)
            for j, uid in enumerate(bu.tolist()):
                pool = neg_pools.get(int(uid))
                if pool is None or int(pool.size) <= 0:
                    bn[j] = int(bp[j])
                else:
                    bn[j] = int(pool[rng.integers(0, int(pool.size))])
            t_user = torch.from_numpy(bu).to(device)
            t_pos = torch.from_numpy(bp).to(device)
            t_neg = torch.from_numpy(bn).to(device)
            t_weight = torch.from_numpy(np.maximum(1e-3, bw)).to(device)
            pos_score, neg_score = model(t_user, t_pos, t_neg)
            pair_loss = -torch.nn.functional.logsigmoid(pos_score - neg_score)
            bpr_loss = (pair_loss * t_weight).mean()
            u = model.user_factors(t_user)
            pi = model.item_factors(t_pos)
            ni = model.item_factors(t_neg)
            reg_loss = float(cfg.reg) * (u.pow(2).sum(dim=1) + pi.pow(2).sum(dim=1) + ni.pow(2).sum(dim=1)).mean()
            loss = bpr_loss + reg_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu().item()))
        history.append(
            {
                "epoch": float(epoch),
                "loss": float(np.mean(losses)) if losses else float("nan"),
            }
        )

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "cfg": asdict(cfg),
            "state_dict": model.state_dict(),
            "history": history,
            "user_to_id": dict(user_to_id),
        },
        str(out),
    )
    return {
        "checkpoint": str(out),
        "history": history,
        "n_examples": int(user_idx.shape[0]),
        "n_users": int(len(user_to_id)),
    }


def load_bpr_mf(path: str | Path, map_location: str | None = None) -> tuple[BPRMF, dict[str, int]]:
    obj = torch.load(str(path), map_location=map_location)
    cfg = BPRMFConfig(**obj["cfg"])
    model = BPRMF(cfg)
    model.load_state_dict(obj["state_dict"])
    user_to_id = {str(k): int(v) for k, v in obj["user_to_id"].items()}
    return model, user_to_id


def recommend_bpr_mf(
    model: BPRMF,
    user_to_id: dict[str, int],
    tracks: Tracks,
    interactions: list[Interaction],
    user_id: str,
    target_culture: str,
    k: int = 20,
    device: torch.device | None = None,
) -> list[Recommendation]:
    if device is None:
        device = torch.device("cpu")
    if str(user_id) not in user_to_id:
        raise ValueError(f"user_id not found in BPR model: {user_id}")
    hist_idx, _, cand_idx = _prepare_user_and_candidates(tracks, interactions, user_id, target_culture)
    user_idx = torch.full(
        (int(cand_idx.shape[0]),),
        int(user_to_id[str(user_id)]),
        dtype=torch.long,
        device=device,
    )
    item_idx = torch.from_numpy(cand_idx.astype(np.int64)).to(device)
    model.eval()
    model.to(device)
    with torch.no_grad():
        scores = model.score(user_idx, item_idx).detach().cpu().numpy().astype(np.float32)
    scores = _safe_softmax(scores)
    return _finalize_embedding_recommendations(tracks, hist_idx, cand_idx, scores, k=k)


def train_content_bpr_mf(
    tracks: Tracks,
    interactions: list[Interaction],
    out_path: str | Path,
    latent_dim: int = 64,
    epochs: int = 12,
    batch_size: int = 512,
    lr: float = 3e-3,
    reg: float = 1e-4,
    profile_weight: float = 0.75,
    content_weight: float = 1.0,
    culture_weight: float = 0.25,
    source_weight: float = 0.15,
    seed: int = 42,
    prefer_cuda: bool = False,
) -> dict[str, object]:
    rng = np.random.default_rng(int(seed))
    track_id_to_idx = _track_id_to_idx(tracks)
    user_to_id, user_idx, pos_idx, weights, neg_pools = _build_bpr_training_state(
        tracks=tracks, interactions=interactions
    )
    if user_idx.size <= 0:
        raise RuntimeError("could not build content-BPR training state")
    user_profiles = _user_content_profiles(
        tracks=tracks,
        interactions=interactions,
        user_to_id=user_to_id,
        track_id_to_idx=track_id_to_idx,
    )
    culture_to_id, source_to_id, item_culture_ids, item_source_ids = _item_feature_id_tables(tracks)

    device = torch.device("cuda" if prefer_cuda and torch.cuda.is_available() else "cpu")
    cfg = ContentBPRMFConfig(
        n_users=int(len(user_to_id)),
        n_items=int(len(tracks)),
        item_feat_dim=int(tracks.embedding.shape[1]),
        n_cultures=int(max(1, len(culture_to_id))),
        n_sources=int(max(1, len(source_to_id))),
        latent_dim=int(latent_dim),
        reg=float(reg),
        profile_weight=float(profile_weight),
        content_weight=float(content_weight),
        culture_weight=float(culture_weight),
        source_weight=float(source_weight),
    )
    model = ContentBPRMF(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr))

    item_features_t = torch.from_numpy(tracks.embedding.astype(np.float32)).to(device)
    item_culture_t = torch.from_numpy(item_culture_ids.astype(np.int64)).to(device)
    item_source_t = torch.from_numpy(item_source_ids.astype(np.int64)).to(device)
    user_profiles_t = torch.from_numpy(user_profiles.astype(np.float32)).to(device)
    order = np.arange(user_idx.shape[0], dtype=np.int64)
    history: list[dict[str, float]] = []
    for epoch in range(int(epochs)):
        rng.shuffle(order)
        losses: list[float] = []
        model.train()
        for start in range(0, int(order.shape[0]), int(batch_size)):
            batch_ids = order[start : start + int(batch_size)]
            bu = user_idx[batch_ids]
            bp = pos_idx[batch_ids]
            bw = weights[batch_ids]
            bn = np.empty_like(bp)
            for j, uid in enumerate(bu.tolist()):
                pool = neg_pools.get(int(uid))
                if pool is None or int(pool.size) <= 0:
                    bn[j] = int(bp[j])
                else:
                    bn[j] = int(pool[rng.integers(0, int(pool.size))])
            t_user = torch.from_numpy(bu).to(device)
            t_pos = torch.from_numpy(bp).to(device)
            t_neg = torch.from_numpy(bn).to(device)
            t_weight = torch.from_numpy(np.maximum(1e-3, bw)).to(device)
            t_profile = user_profiles_t[t_user]
            pos_score, neg_score = model(
                user_idx=t_user,
                pos_idx=t_pos,
                neg_idx=t_neg,
                user_profile=t_profile,
                item_features=item_features_t,
                item_culture_ids=item_culture_t,
                item_source_ids=item_source_t,
            )
            pair_loss = -torch.nn.functional.logsigmoid(pos_score - neg_score)
            bpr_loss = (pair_loss * t_weight).mean()
            reg_loss = (
                float(cfg.reg)
                * (
                    model.user_factors(t_user).pow(2).sum(dim=1)
                    + model.item_factors(t_pos).pow(2).sum(dim=1)
                    + model.item_factors(t_neg).pow(2).sum(dim=1)
                ).mean()
            )
            proj_reg = float(cfg.reg) * (
                model.user_proj.weight.pow(2).mean()
                + model.item_proj.weight.pow(2).mean()
                + model.culture_factors.weight.pow(2).mean()
                + model.source_factors.weight.pow(2).mean()
            )
            loss = bpr_loss + reg_loss + proj_reg
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu().item()))
        history.append(
            {
                "epoch": float(epoch),
                "loss": float(np.mean(losses)) if losses else float("nan"),
            }
        )

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "cfg": asdict(cfg),
            "state_dict": model.state_dict(),
            "history": history,
            "user_to_id": dict(user_to_id),
            "culture_to_id": dict(culture_to_id),
            "source_to_id": dict(source_to_id),
        },
        str(out),
    )
    return {
        "checkpoint": str(out),
        "history": history,
        "n_examples": int(user_idx.shape[0]),
        "n_users": int(len(user_to_id)),
    }


def load_content_bpr_mf(
    path: str | Path,
    map_location: str | None = None,
) -> tuple[ContentBPRMF, dict[str, int], dict[str, int], dict[str, int]]:
    obj = torch.load(str(path), map_location=map_location)
    cfg = ContentBPRMFConfig(**obj["cfg"])
    model = ContentBPRMF(cfg)
    model.load_state_dict(obj["state_dict"])
    user_to_id = {str(k): int(v) for k, v in obj["user_to_id"].items()}
    culture_to_id = {str(k): int(v) for k, v in obj["culture_to_id"].items()}
    source_to_id = {str(k): int(v) for k, v in obj["source_to_id"].items()}
    return model, user_to_id, culture_to_id, source_to_id


def recommend_content_bpr_mf(
    model: ContentBPRMF,
    user_to_id: dict[str, int],
    culture_to_id: dict[str, int],
    source_to_id: dict[str, int],
    tracks: Tracks,
    interactions: list[Interaction],
    user_id: str,
    target_culture: str,
    k: int = 20,
    mf_weight: float = 0.6,
    content_weight: float = 0.2,
    novelty_weight: float = 0.1,
    minority_weight: float = 0.05,
    source_weight: float = 0.05,
    device: torch.device | None = None,
) -> list[Recommendation]:
    if device is None:
        device = torch.device("cpu")
    if str(user_id) not in user_to_id:
        raise ValueError(f"user_id not found in content BPR model: {user_id}")
    hist_idx, hist_w, cand_idx = _prepare_user_and_candidates(tracks, interactions, user_id, target_culture)
    user_profile = _user_profile(tracks.embedding, hist_idx, hist_w).astype(np.float32)
    user_profile_t = torch.from_numpy(np.repeat(user_profile.reshape(1, -1), int(cand_idx.shape[0]), axis=0)).to(device)
    item_features_t = torch.from_numpy(tracks.embedding.astype(np.float32)).to(device)
    item_culture_ids = np.array(
        [int(culture_to_id.get(str(x), 0)) for x in tracks.culture.tolist()],
        dtype=np.int64,
    )
    if tracks.source_dataset is None:
        item_source_ids = np.zeros((len(tracks),), dtype=np.int64)
    else:
        item_source_ids = np.array(
            [int(source_to_id.get(str(x), 0)) for x in tracks.source_dataset.tolist()],
            dtype=np.int64,
        )
    item_culture_t = torch.from_numpy(item_culture_ids).to(device)
    item_source_t = torch.from_numpy(item_source_ids).to(device)
    user_idx = torch.full(
        (int(cand_idx.shape[0]),),
        int(user_to_id[str(user_id)]),
        dtype=torch.long,
        device=device,
    )
    item_idx = torch.from_numpy(cand_idx.astype(np.int64)).to(device)
    model.eval()
    model.to(device)
    with torch.no_grad():
        scores = (
            model.score_with_features(
                user_idx=user_idx,
                item_idx=item_idx,
                user_profile=user_profile_t,
                item_features=item_features_t,
                item_culture_ids=item_culture_t,
                item_source_ids=item_source_t,
            )
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )
    mf_scores = _minmax(scores)
    cand_emb = tracks.embedding[cand_idx]
    hist_emb = tracks.embedding[hist_idx]
    content_scores = _minmax(_cosine_similarity(user_profile, cand_emb))
    novelty_scores = _minmax(np.linalg.norm(hist_emb[:, None, :] - cand_emb[None, :, :], axis=2).mean(axis=0))
    pop_by_track = _popularity_by_track(interactions)
    minority_raw = np.array(
        [float(-pop_by_track.get(str(tracks.track_id[i]), 0.0)) for i in cand_idx.tolist()],
        dtype=np.float32,
    )
    minority_scores = _minmax(minority_raw)
    if tracks.source_dataset is None:
        source_scores = np.zeros_like(mf_scores, dtype=np.float32)
    else:
        src_values = [str(x) for x in tracks.source_dataset.tolist()]
        unique, counts = np.unique(np.array(src_values, dtype=object), return_counts=True)
        source_inv = {str(k): 1.0 / float(v) for k, v in zip(unique.tolist(), counts.tolist())}
        source_scores = _minmax(
            np.array(
                [float(source_inv.get(str(tracks.source_dataset[int(i)]), 0.0)) for i in cand_idx.tolist()],
                dtype=np.float32,
            )
        )
    final_scores = (
        float(mf_weight) * mf_scores
        + float(content_weight) * content_scores
        + float(novelty_weight) * novelty_scores
        + float(minority_weight) * minority_scores
        + float(source_weight) * source_scores
    ).astype(np.float32)
    final_scores = _safe_softmax(final_scores)
    return _finalize_embedding_recommendations(tracks, hist_idx, cand_idx, final_scores, k=k)


def _bpr_scores_for_candidates(
    model: BPRMF,
    user_to_id: dict[str, int],
    user_id: str,
    cand_idx: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    if str(user_id) not in user_to_id:
        raise ValueError(f"user_id not found in BPR model: {user_id}")
    user_idx = torch.full(
        (int(cand_idx.shape[0]),),
        int(user_to_id[str(user_id)]),
        dtype=torch.long,
        device=device,
    )
    item_idx = torch.from_numpy(cand_idx.astype(np.int64)).to(device)
    with torch.no_grad():
        scores = model.score(user_idx, item_idx).detach().cpu().numpy().astype(np.float32)
    return scores


def _bpr_hybrid_feature_table(
    tracks: Tracks,
    interactions: list[Interaction],
    hist_idx: np.ndarray,
    hist_w: np.ndarray,
    cand_idx: np.ndarray,
    target_culture: str,
    pop_by_track: dict[str, float],
    source_inv_freq: dict[str, float],
    culture_centroids: dict[str, np.ndarray],
    bpr_model: BPRMF,
    user_to_id: dict[str, int],
    user_id: str,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    feature_mat, _ = _candidate_feature_table(
        tracks=tracks,
        interactions=interactions,
        hist_idx=hist_idx,
        hist_w=hist_w,
        cand_idx=cand_idx,
        target_culture=target_culture,
        pop_by_track=pop_by_track,
        source_inv_freq=source_inv_freq,
        culture_centroids=culture_centroids,
    )
    bpr_scores = _minmax(_bpr_scores_for_candidates(bpr_model, user_to_id, user_id, cand_idx, device))
    recall_score = (
        0.38 * bpr_scores
        + 0.15 * feature_mat[:, 0]
        + 0.12 * feature_mat[:, 1]
        + 0.08 * feature_mat[:, 2]
        + 0.07 * feature_mat[:, 3]
        + 0.08 * feature_mat[:, 4]
        + 0.05 * feature_mat[:, 6]
        + 0.05 * feature_mat[:, 7]
        + 0.02 * feature_mat[:, 8]
    ).astype(np.float32)
    scalar_features = np.concatenate(
        [
            feature_mat.astype(np.float32),
            bpr_scores.reshape(-1, 1),
            recall_score.reshape(-1, 1),
        ],
        axis=1,
    ).astype(np.float32)
    return scalar_features, recall_score


def _make_bpr_hybrid_pairwise_examples(
    tracks: Tracks,
    interactions: list[Interaction],
    bpr_model: BPRMF,
    user_to_id: dict[str, int],
    recall_k: int,
    negative_samples: int,
    hard_negative_ratio: float,
    seed: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    track_id_to_idx = _track_id_to_idx(tracks)
    by_user: dict[str, list[Interaction]] = {}
    for it in interactions:
        by_user.setdefault(str(it.user_id), []).append(it)

    pop_by_track = _popularity_by_track(interactions)
    source_inv_freq = _source_inverse_frequency(tracks)
    culture_centroids = _culture_centroid_embeddings(tracks)

    pos_features: list[np.ndarray] = []
    neg_features: list[np.ndarray] = []
    pair_weights: list[float] = []
    for user_id in sorted(by_user.keys()):
        if str(user_id) not in user_to_id:
            continue
        rows = [it for it in by_user[user_id] if str(it.track_id) in track_id_to_idx]
        if len(rows) <= 1:
            continue
        pos_idx = np.array([track_id_to_idx[str(it.track_id)] for it in rows], dtype=np.int64)
        weights = np.array([float(it.weight) for it in rows], dtype=np.float32)
        weights = weights / max(1e-12, float(weights.sum()))
        seen = {int(i) for i in pos_idx.tolist()}
        for j, pos in enumerate(pos_idx.tolist()):
            mask = np.ones_like(pos_idx, dtype=bool)
            mask[int(j)] = False
            context_idx = pos_idx[mask]
            context_w = weights[mask]
            if int(context_idx.size) <= 0:
                continue
            context_w = context_w / max(1e-12, float(context_w.sum()))
            target_culture = str(tracks.culture[int(pos)])
            cand_all = tracks.indices_of_cultures([target_culture]).astype(np.int64)
            cand_idx = np.array(
                [int(i) for i in cand_all.tolist() if int(i) == int(pos) or int(i) not in seen],
                dtype=np.int64,
            )
            if int(cand_idx.size) <= 1:
                continue
            user_vec = _user_profile(tracks.embedding, context_idx, context_w)
            scalar_features, recall_score = _bpr_hybrid_feature_table(
                tracks=tracks,
                interactions=interactions,
                hist_idx=context_idx,
                hist_w=context_w,
                cand_idx=cand_idx,
                target_culture=target_culture,
                pop_by_track=pop_by_track,
                source_inv_freq=source_inv_freq,
                culture_centroids=culture_centroids,
                bpr_model=bpr_model,
                user_to_id=user_to_id,
                user_id=str(user_id),
                device=device,
            )
            model_feature_mat = _hybrid_model_feature_matrix(
                user_vec=user_vec,
                cand_emb=tracks.embedding[cand_idx],
                scalar_features=scalar_features,
            )
            pos_local = np.nonzero(cand_idx == int(pos))[0]
            if int(pos_local.size) <= 0:
                continue
            pos_local_idx = int(pos_local[0])
            ordered = np.argsort(-recall_score)
            ordered = ordered[ordered != pos_local_idx]
            if int(ordered.size) <= 0:
                continue
            hard_limit = min(int(recall_k), int(ordered.size))
            hard_pool = ordered[:hard_limit]
            hard_count = min(
                int(round(float(negative_samples) * float(hard_negative_ratio))),
                int(hard_pool.size),
            )
            easy_count = max(0, int(negative_samples) - int(hard_count))
            chosen: list[int] = []
            if hard_count > 0 and int(hard_pool.size) > 0:
                chosen.extend(rng.choice(hard_pool, size=hard_count, replace=False).tolist())
            if easy_count > 0:
                easy_pool = np.array(
                    [int(x) for x in ordered.tolist() if int(x) not in set(chosen)],
                    dtype=np.int64,
                )
                if int(easy_pool.size) > 0:
                    take = min(int(easy_count), int(easy_pool.size))
                    chosen.extend(rng.choice(easy_pool, size=take, replace=False).tolist())
            if not chosen:
                chosen = ordered[: min(int(negative_samples), int(ordered.size))].tolist()
            for neg_local_idx in chosen:
                pos_features.append(model_feature_mat[pos_local_idx].astype(np.float32))
                neg_features.append(model_feature_mat[int(neg_local_idx)].astype(np.float32))
                pair_weights.append(float(weights[int(j)]))

    if not pos_features:
        raise RuntimeError("could not build BPR-hybrid pairwise training examples")
    return (
        np.stack(pos_features, axis=0).astype(np.float32),
        np.stack(neg_features, axis=0).astype(np.float32),
        np.array(pair_weights, dtype=np.float32),
    )


def _make_bpr_hybrid_listwise_queries(
    tracks: Tracks,
    interactions: list[Interaction],
    bpr_model: BPRMF,
    user_to_id: dict[str, int],
    recall_k: int,
    seed: int,
    device: torch.device,
) -> list[dict[str, object]]:
    _ = int(seed)
    track_id_to_idx = _track_id_to_idx(tracks)
    by_user: dict[str, list[Interaction]] = {}
    for it in interactions:
        by_user.setdefault(str(it.user_id), []).append(it)

    pop_by_track = _popularity_by_track(interactions)
    source_inv_freq = _source_inverse_frequency(tracks)
    culture_centroids = _culture_centroid_embeddings(tracks)

    queries: list[dict[str, object]] = []
    for user_id in sorted(by_user.keys()):
        if str(user_id) not in user_to_id:
            continue
        rows = [it for it in by_user[user_id] if str(it.track_id) in track_id_to_idx]
        if len(rows) <= 1:
            continue
        pos_idx = np.array([track_id_to_idx[str(it.track_id)] for it in rows], dtype=np.int64)
        weights = np.array([float(it.weight) for it in rows], dtype=np.float32)
        weights = weights / max(1e-12, float(weights.sum()))
        seen = {int(i) for i in pos_idx.tolist()}
        for j, pos in enumerate(pos_idx.tolist()):
            mask = np.ones_like(pos_idx, dtype=bool)
            mask[int(j)] = False
            context_idx = pos_idx[mask]
            context_w = weights[mask]
            if int(context_idx.size) <= 0:
                continue
            context_w = context_w / max(1e-12, float(context_w.sum()))
            target_culture = str(tracks.culture[int(pos)])
            cand_all = tracks.indices_of_cultures([target_culture]).astype(np.int64)
            cand_idx = np.array(
                [int(i) for i in cand_all.tolist() if int(i) == int(pos) or int(i) not in seen],
                dtype=np.int64,
            )
            if int(cand_idx.size) <= 1:
                continue
            user_vec = _user_profile(tracks.embedding, context_idx, context_w)
            scalar_features, recall_score = _bpr_hybrid_feature_table(
                tracks=tracks,
                interactions=interactions,
                hist_idx=context_idx,
                hist_w=context_w,
                cand_idx=cand_idx,
                target_culture=target_culture,
                pop_by_track=pop_by_track,
                source_inv_freq=source_inv_freq,
                culture_centroids=culture_centroids,
                bpr_model=bpr_model,
                user_to_id=user_to_id,
                user_id=str(user_id),
                device=device,
            )
            pos_local = np.nonzero(cand_idx == int(pos))[0]
            if int(pos_local.size) <= 0:
                continue
            pos_local_idx = int(pos_local[0])
            ordered = np.argsort(-recall_score)
            recall_n = min(max(20, int(recall_k)), int(cand_idx.shape[0]))
            selected_local = ordered[:recall_n].tolist()
            if pos_local_idx not in selected_local:
                if selected_local:
                    selected_local[-1] = pos_local_idx
                else:
                    selected_local = [pos_local_idx]
            selected_local_arr = np.array(selected_local, dtype=np.int64)
            cand_idx_recall = cand_idx[selected_local_arr].astype(np.int64)
            pos_in_recall = np.nonzero(cand_idx_recall == int(pos))[0]
            if int(pos_in_recall.size) <= 0:
                continue
            queries.append(
                {
                    "user_vec": user_vec.astype(np.float32),
                    "cand_idx": cand_idx_recall,
                    "scalar_features": scalar_features[selected_local_arr].astype(np.float32),
                    "pos_local": int(pos_in_recall[0]),
                    "query_weight": float(weights[int(j)]),
                }
            )
    if not queries:
        raise RuntimeError("could not build BPR-hybrid listwise training queries")
    return queries


def train_bpr_two_stage_hybrid_ranker(
    tracks: Tracks,
    interactions: list[Interaction],
    bpr_checkpoint: str | Path,
    out_path: str | Path,
    hidden_dim: int = 256,
    depth: int = 3,
    dropout: float = 0.1,
    epochs: int = 6,
    batch_size: int = 256,
    lr: float = 1e-3,
    recall_k: int = 100,
    negative_samples: int = 6,
    hard_negative_ratio: float = 0.8,
    seed: int = 42,
    prefer_cuda: bool = False,
) -> dict[str, object]:
    device = torch.device("cuda" if prefer_cuda and torch.cuda.is_available() else "cpu")
    bpr_model, user_to_id = load_bpr_mf(bpr_checkpoint, map_location=str(device))
    bpr_model.to(device)
    bpr_model.eval()
    pos_x, neg_x, pair_w = _make_bpr_hybrid_pairwise_examples(
        tracks=tracks,
        interactions=interactions,
        bpr_model=bpr_model,
        user_to_id=user_to_id,
        recall_k=int(recall_k),
        negative_samples=int(negative_samples),
        hard_negative_ratio=float(hard_negative_ratio),
        seed=int(seed),
        device=device,
    )
    ds = TensorDataset(torch.from_numpy(pos_x), torch.from_numpy(neg_x), torch.from_numpy(pair_w))
    dl = DataLoader(ds, batch_size=min(int(batch_size), len(ds)), shuffle=True, drop_last=False)
    cfg = TwoStageHybridConfig(
        input_dim=int(pos_x.shape[1]),
        hidden_dim=int(hidden_dim),
        depth=int(depth),
        dropout=float(dropout),
    )
    model = TwoStageHybridRanker(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr))
    history: list[dict[str, float]] = []
    for epoch in range(int(epochs)):
        model.train()
        losses: list[float] = []
        for batch_pos, batch_neg, batch_w in dl:
            batch_pos = batch_pos.to(device)
            batch_neg = batch_neg.to(device)
            batch_w = batch_w.to(device)
            pos_logits = model(batch_pos)
            neg_logits = model(batch_neg)
            pair_loss = -torch.nn.functional.logsigmoid(pos_logits - neg_logits)
            loss = (pair_loss * batch_w).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu().item()))
        history.append(
            {
                "epoch": float(epoch),
                "loss": float(np.mean(losses)) if losses else float("nan"),
            }
        )

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "cfg": asdict(cfg),
            "state_dict": model.state_dict(),
            "history": history,
            "bpr_checkpoint": str(bpr_checkpoint),
        },
        str(out),
    )
    return {"checkpoint": str(out), "history": history, "n_pairs": int(pos_x.shape[0])}


def train_bpr_listwise_hybrid_ranker(
    tracks: Tracks,
    interactions: list[Interaction],
    bpr_checkpoint: str | Path,
    out_path: str | Path,
    hidden_dim: int = 256,
    depth: int = 3,
    dropout: float = 0.1,
    epochs: int = 4,
    lr: float = 5e-4,
    recall_k: int = 120,
    warm_start_checkpoint: str | Path | None = None,
    seed: int = 42,
    prefer_cuda: bool = False,
) -> dict[str, object]:
    device = torch.device("cuda" if prefer_cuda and torch.cuda.is_available() else "cpu")
    bpr_model, user_to_id = load_bpr_mf(bpr_checkpoint, map_location=str(device))
    bpr_model.to(device)
    bpr_model.eval()
    queries = _make_bpr_hybrid_listwise_queries(
        tracks=tracks,
        interactions=interactions,
        bpr_model=bpr_model,
        user_to_id=user_to_id,
        recall_k=int(recall_k),
        seed=int(seed),
        device=device,
    )
    first_q = queries[0]
    first_feat = _hybrid_model_feature_matrix(
        user_vec=np.asarray(first_q["user_vec"], dtype=np.float32),
        cand_emb=tracks.embedding[np.asarray(first_q["cand_idx"], dtype=np.int64)],
        scalar_features=np.asarray(first_q["scalar_features"], dtype=np.float32),
    )
    cfg = TwoStageHybridConfig(
        input_dim=int(first_feat.shape[1]),
        hidden_dim=int(hidden_dim),
        depth=int(depth),
        dropout=float(dropout),
    )
    model = TwoStageHybridRanker(cfg).to(device)
    if warm_start_checkpoint is not None and Path(warm_start_checkpoint).exists():
        warm = torch.load(str(warm_start_checkpoint), map_location=str(device))
        state_dict = warm.get("state_dict")
        if isinstance(state_dict, dict):
            model.load_state_dict(state_dict, strict=False)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr))
    rng = np.random.default_rng(int(seed))
    history: list[dict[str, float]] = []
    for epoch in range(int(epochs)):
        model.train()
        losses: list[float] = []
        for q_idx in rng.permutation(len(queries)).tolist():
            query = queries[int(q_idx)]
            feat = _hybrid_model_feature_matrix(
                user_vec=np.asarray(query["user_vec"], dtype=np.float32),
                cand_emb=tracks.embedding[np.asarray(query["cand_idx"], dtype=np.int64)],
                scalar_features=np.asarray(query["scalar_features"], dtype=np.float32),
            )
            logits = model(torch.from_numpy(feat).to(device))
            target_idx = int(query["pos_local"])
            weight = float(query["query_weight"])
            loss = -torch.log_softmax(logits, dim=0)[target_idx] * weight
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu().item()))
        history.append(
            {
                "epoch": float(epoch),
                "loss": float(np.mean(losses)) if losses else float("nan"),
            }
        )

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "cfg": asdict(cfg),
            "state_dict": model.state_dict(),
            "history": history,
            "bpr_checkpoint": str(bpr_checkpoint),
            "warm_start_checkpoint": str(warm_start_checkpoint) if warm_start_checkpoint is not None else None,
        },
        str(out),
    )
    return {"checkpoint": str(out), "history": history, "n_queries": int(len(queries))}


def load_bpr_two_stage_hybrid_ranker(path: str | Path, map_location: str | None = None) -> TwoStageHybridRanker:
    obj = torch.load(str(path), map_location=map_location)
    cfg = TwoStageHybridConfig(**obj["cfg"])
    model = TwoStageHybridRanker(cfg)
    model.load_state_dict(obj["state_dict"])
    return model


def load_bpr_listwise_hybrid_ranker(path: str | Path, map_location: str | None = None) -> TwoStageHybridRanker:
    return load_bpr_two_stage_hybrid_ranker(path=path, map_location=map_location)


def _build_bpr_tree_ranker_dataset(
    tracks: Tracks,
    queries: list[dict[str, object]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    feature_blocks: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    groups: list[int] = []
    sample_weights: list[np.ndarray] = []
    for query in queries:
        _ = np.asarray(query["cand_idx"], dtype=np.int64)
        feat = np.asarray(query["scalar_features"], dtype=np.float32)
        y = np.zeros((int(feat.shape[0]),), dtype=np.float32)
        y[int(query["pos_local"])] = 1.0
        q_weight = float(query["query_weight"])
        feature_blocks.append(feat)
        labels.append(y)
        groups.append(int(feat.shape[0]))
        sample_weights.append(np.full((int(feat.shape[0]),), q_weight, dtype=np.float32))
    if not feature_blocks:
        raise RuntimeError("could not build tree ranker dataset")
    return (
        np.concatenate(feature_blocks, axis=0).astype(np.float32),
        np.concatenate(labels, axis=0).astype(np.float32),
        np.array(groups, dtype=np.int32),
        np.concatenate(sample_weights, axis=0).astype(np.float32),
    )


def train_bpr_tree_hybrid_ranker(
    tracks: Tracks,
    interactions: list[Interaction],
    bpr_checkpoint: str | Path,
    out_path: str | Path,
    backend: str = "lightgbm",
    recall_k: int = 120,
    n_estimators: int = 300,
    learning_rate: float = 0.05,
    num_leaves: int = 31,
    max_depth: int = -1,
    min_child_samples: int = 20,
    subsample: float = 0.9,
    colsample_bytree: float = 0.8,
    reg_lambda: float = 1.0,
    seed: int = 42,
    prefer_cuda: bool = False,
) -> dict[str, object]:
    device = torch.device("cuda" if prefer_cuda and torch.cuda.is_available() else "cpu")
    bpr_model, user_to_id = load_bpr_mf(bpr_checkpoint, map_location=str(device))
    bpr_model.to(device)
    bpr_model.eval()
    queries = _make_bpr_hybrid_listwise_queries(
        tracks=tracks,
        interactions=interactions,
        bpr_model=bpr_model,
        user_to_id=user_to_id,
        recall_k=int(recall_k),
        seed=int(seed),
        device=device,
    )
    x, y, groups, sample_weights = _build_bpr_tree_ranker_dataset(tracks=tracks, queries=queries)
    backend_key = str(backend).strip().lower()
    model: object
    if backend_key == "lightgbm":
        import lightgbm as lgb

        model = lgb.LGBMRanker(
            objective="lambdarank",
            metric="ndcg",
            boosting_type="gbdt",
            n_estimators=int(n_estimators),
            learning_rate=float(learning_rate),
            num_leaves=int(num_leaves),
            max_depth=int(max_depth),
            min_child_samples=int(min_child_samples),
            subsample=float(subsample),
            colsample_bytree=float(colsample_bytree),
            reg_lambda=float(reg_lambda),
            random_state=int(seed),
            verbosity=-1,
        )
        model.fit(
            x,
            y,
            group=groups.tolist(),
            sample_weight=sample_weights,
        )
    elif backend_key == "xgboost":
        import xgboost as xgb

        model = xgb.XGBRanker(
            objective="rank:ndcg",
            tree_method="hist",
            n_estimators=int(n_estimators),
            learning_rate=float(learning_rate),
            max_depth=6 if int(max_depth) <= 0 else int(max_depth),
            min_child_weight=max(1.0, float(min_child_samples) / 4.0),
            subsample=float(subsample),
            colsample_bytree=float(colsample_bytree),
            reg_lambda=float(reg_lambda),
            random_state=int(seed),
        )
        model.fit(
            x,
            y,
            group=groups.tolist(),
            sample_weight=sample_weights,
            verbose=False,
        )
    elif backend_key == "sklearn_histgbdt":
        from sklearn.ensemble import HistGradientBoostingRegressor

        model = HistGradientBoostingRegressor(
            learning_rate=float(learning_rate),
            max_iter=int(n_estimators),
            max_leaf_nodes=max(3, int(num_leaves)),
            max_depth=None if int(max_depth) <= 0 else int(max_depth),
            min_samples_leaf=max(1, int(min_child_samples)),
            l2_regularization=float(reg_lambda),
            random_state=int(seed),
        )
        model.fit(x, y, sample_weight=sample_weights)
    else:
        raise ValueError(f"unsupported tree ranker backend: {backend}")

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(
            {
                "backend": backend_key,
                "model": model,
                "bpr_checkpoint": str(bpr_checkpoint),
                "history": {
                    "n_queries": int(len(queries)),
                    "n_rows": int(x.shape[0]),
                    "feature_dim": int(x.shape[1]),
                    "backend": backend_key,
                },
            },
            f,
        )
    return {
        "checkpoint": str(out),
        "backend": backend_key,
        "n_queries": int(len(queries)),
        "n_rows": int(x.shape[0]),
        "feature_dim": int(x.shape[1]),
    }


def load_bpr_tree_hybrid_ranker(path: str | Path) -> dict[str, object]:
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, dict) or "model" not in obj:
        raise ValueError(f"invalid tree ranker checkpoint: {path}")
    return obj


def _predict_tree_ranker_scores(ranker_obj: dict[str, object], features: np.ndarray) -> np.ndarray:
    backend = str(ranker_obj.get("backend", "lightgbm"))
    model = ranker_obj["model"]
    if backend in {"lightgbm", "xgboost"}:
        scores = np.asarray(model.predict(features), dtype=np.float32)
    elif backend == "sklearn_histgbdt":
        scores = np.asarray(model.predict(features), dtype=np.float32)
    else:
        raise ValueError(f"unsupported tree ranker backend: {backend}")
    return scores.reshape(-1).astype(np.float32)


def recommend_embedding_bpr_tree_hybrid(
    ranker_obj: dict[str, object],
    bpr_model: BPRMF,
    user_to_id: dict[str, int],
    tracks: Tracks,
    interactions: list[Interaction],
    user_id: str,
    target_culture: str,
    k: int = 20,
    recall_k: int = 120,
    rerank_weight: float = 0.58,
    recall_weight: float = 0.12,
    bpr_weight: float = 0.08,
    novelty_weight: float = 0.02,
    target_affinity_weight: float = 0.12,
    minority_weight: float = 0.06,
    source_weight: float = 0.02,
    device: torch.device | None = None,
) -> list[Recommendation]:
    if device is None:
        device = torch.device("cpu")
    hist_idx, hist_w, cand_idx = _prepare_user_and_candidates(tracks, interactions, user_id, target_culture)
    _user_profile(tracks.embedding, hist_idx, hist_w)
    pop_by_track = _popularity_by_track(interactions)
    source_inv_freq = _source_inverse_frequency(tracks)
    culture_centroids = _culture_centroid_embeddings(tracks)
    scalar_features, recall_score = _bpr_hybrid_feature_table(
        tracks=tracks,
        interactions=interactions,
        hist_idx=hist_idx,
        hist_w=hist_w,
        cand_idx=cand_idx,
        target_culture=target_culture,
        pop_by_track=pop_by_track,
        source_inv_freq=source_inv_freq,
        culture_centroids=culture_centroids,
        bpr_model=bpr_model,
        user_to_id=user_to_id,
        user_id=str(user_id),
        device=device,
    )
    recall_n = min(max(int(k), int(recall_k)), int(cand_idx.shape[0]))
    recall_local = np.argsort(-recall_score)[:recall_n]
    cand_idx_recall = cand_idx[recall_local]
    feat_recall = scalar_features[recall_local].astype(np.float32)
    rerank_scores = _minmax(_predict_tree_ranker_scores(ranker_obj, feat_recall))
    recall_scores = _minmax(recall_score[recall_local])
    bpr_scores = scalar_features[recall_local, 10]
    novelty_scores = scalar_features[recall_local, 4]
    target_scores = scalar_features[recall_local, 7]
    minority_scores = scalar_features[recall_local, 6]
    source_scores = scalar_features[recall_local, 9]
    final_scores = (
        float(rerank_weight) * rerank_scores
        + float(recall_weight) * recall_scores
        + float(bpr_weight) * bpr_scores
        + float(novelty_weight) * novelty_scores
        + float(target_affinity_weight) * target_scores
        + float(minority_weight) * minority_scores
        + float(source_weight) * source_scores
    ).astype(np.float32)
    return _finalize_embedding_recommendations(tracks, hist_idx, cand_idx_recall, final_scores, k=k)


def recommend_embedding_bpr_two_stage_hybrid(
    ranker: TwoStageHybridRanker,
    bpr_model: BPRMF,
    user_to_id: dict[str, int],
    tracks: Tracks,
    interactions: list[Interaction],
    user_id: str,
    target_culture: str,
    k: int = 20,
    recall_k: int = 120,
    rerank_weight: float = 0.62,
    recall_weight: float = 0.16,
    bpr_weight: float = 0.10,
    novelty_weight: float = 0.0,
    target_affinity_weight: float = 0.08,
    minority_weight: float = 0.02,
    source_weight: float = 0.02,
    device: torch.device | None = None,
) -> list[Recommendation]:
    if device is None:
        device = torch.device("cpu")
    hist_idx, hist_w, cand_idx = _prepare_user_and_candidates(tracks, interactions, user_id, target_culture)
    user_vec = _user_profile(tracks.embedding, hist_idx, hist_w)
    pop_by_track = _popularity_by_track(interactions)
    source_inv_freq = _source_inverse_frequency(tracks)
    culture_centroids = _culture_centroid_embeddings(tracks)
    scalar_features, recall_score = _bpr_hybrid_feature_table(
        tracks=tracks,
        interactions=interactions,
        hist_idx=hist_idx,
        hist_w=hist_w,
        cand_idx=cand_idx,
        target_culture=target_culture,
        pop_by_track=pop_by_track,
        source_inv_freq=source_inv_freq,
        culture_centroids=culture_centroids,
        bpr_model=bpr_model,
        user_to_id=user_to_id,
        user_id=str(user_id),
        device=device,
    )
    recall_n = min(max(int(k), int(recall_k)), int(cand_idx.shape[0]))
    recall_local = np.argsort(-recall_score)[:recall_n]
    cand_idx_recall = cand_idx[recall_local]
    feat_recall = _hybrid_model_feature_matrix(
        user_vec=user_vec,
        cand_emb=tracks.embedding[cand_idx_recall],
        scalar_features=scalar_features[recall_local],
    )
    ranker.eval()
    ranker.to(device)
    with torch.no_grad():
        logits = ranker(torch.from_numpy(feat_recall).to(device)).detach().cpu().numpy().astype(np.float32)
    rerank_scores = _minmax(1.0 / (1.0 + np.exp(-logits)))
    recall_scores = _minmax(recall_score[recall_local])
    bpr_scores = scalar_features[recall_local, 10]
    novelty_scores = scalar_features[recall_local, 4]
    target_scores = scalar_features[recall_local, 7]
    minority_scores = scalar_features[recall_local, 6]
    source_scores = scalar_features[recall_local, 9]
    final_scores = (
        float(rerank_weight) * rerank_scores
        + float(recall_weight) * recall_scores
        + float(bpr_weight) * bpr_scores
        + float(novelty_weight) * novelty_scores
        + float(target_affinity_weight) * target_scores
        + float(minority_weight) * minority_scores
        + float(source_weight) * source_scores
    ).astype(np.float32)
    return _finalize_embedding_recommendations(tracks, hist_idx, cand_idx_recall, final_scores, k=k)


def recommend_embedding_bpr_listwise_hybrid(
    ranker: TwoStageHybridRanker,
    bpr_model: BPRMF,
    user_to_id: dict[str, int],
    tracks: Tracks,
    interactions: list[Interaction],
    user_id: str,
    target_culture: str,
    k: int = 20,
    recall_k: int = 120,
    rerank_weight: float = 0.62,
    recall_weight: float = 0.14,
    bpr_weight: float = 0.10,
    novelty_weight: float = 0.02,
    target_affinity_weight: float = 0.06,
    minority_weight: float = 0.04,
    source_weight: float = 0.02,
    device: torch.device | None = None,
) -> list[Recommendation]:
    if device is None:
        device = torch.device("cpu")
    hist_idx, hist_w, cand_idx = _prepare_user_and_candidates(tracks, interactions, user_id, target_culture)
    user_vec = _user_profile(tracks.embedding, hist_idx, hist_w)
    pop_by_track = _popularity_by_track(interactions)
    source_inv_freq = _source_inverse_frequency(tracks)
    culture_centroids = _culture_centroid_embeddings(tracks)
    scalar_features, recall_score = _bpr_hybrid_feature_table(
        tracks=tracks,
        interactions=interactions,
        hist_idx=hist_idx,
        hist_w=hist_w,
        cand_idx=cand_idx,
        target_culture=target_culture,
        pop_by_track=pop_by_track,
        source_inv_freq=source_inv_freq,
        culture_centroids=culture_centroids,
        bpr_model=bpr_model,
        user_to_id=user_to_id,
        user_id=str(user_id),
        device=device,
    )
    recall_n = min(max(int(k), int(recall_k)), int(cand_idx.shape[0]))
    recall_local = np.argsort(-recall_score)[:recall_n]
    cand_idx_recall = cand_idx[recall_local]
    feat_recall = _hybrid_model_feature_matrix(
        user_vec=user_vec,
        cand_emb=tracks.embedding[cand_idx_recall],
        scalar_features=scalar_features[recall_local],
    )
    ranker.eval()
    ranker.to(device)
    with torch.no_grad():
        logits = ranker(torch.from_numpy(feat_recall).to(device)).detach().cpu().numpy().astype(np.float32)
    rerank_scores = _minmax(logits)
    recall_scores = _minmax(recall_score[recall_local])
    bpr_scores = scalar_features[recall_local, 10]
    novelty_scores = scalar_features[recall_local, 4]
    target_scores = scalar_features[recall_local, 7]
    minority_scores = scalar_features[recall_local, 6]
    source_scores = scalar_features[recall_local, 9]
    final_scores = (
        float(rerank_weight) * rerank_scores
        + float(recall_weight) * recall_scores
        + float(bpr_weight) * bpr_scores
        + float(novelty_weight) * novelty_scores
        + float(target_affinity_weight) * target_scores
        + float(minority_weight) * minority_scores
        + float(source_weight) * source_scores
    ).astype(np.float32)
    return _finalize_embedding_recommendations(tracks, hist_idx, cand_idx_recall, final_scores, k=k)
