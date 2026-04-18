from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dcas.data.batch import collate_batch
from dcas.data.interactions import Interaction, load_interactions
from dcas.data.npz_tracks import Tracks, load_tracks
from dcas.data.torch_dataset import CultureVocab, SourceVocab, TrackDataset, make_source_balanced_sampler
from dcas.models.dcas_vae import DCASConfig, DCASModel
from dcas.pal.constraints import PairwiseConstraint, load_constraints
from dcas.pal.uncertainty import rank_by_uncertainty
from dcas.pal.wording import PAL_TASK_QUESTION_ZH
from dcas.recommender import Recommendation, recommend_ot
from dcas.scripts.build_tracks_from_audio import build_tracks_from_audio
from dcas.scripts.make_toy_data import generate_toy_data
from dcas.serialization import load_checkpoint, save_checkpoint
from dcas.style_transfer import generate_counterfactual_embedding
from dcas.utils import get_device, set_seed
from dcas.waveform_style_transfer import transfer_waveform_style


@dataclass(frozen=True)
class RankExample:
    user_id: str
    pos_idx: int
    pos_culture: str
    context_idx: tuple[int, ...]
    context_w: tuple[float, ...]
    weight: float


def _stage_scale(epoch: int, start_epoch: int, warmup_epochs: int) -> float:
    if int(epoch) < int(start_epoch):
        return 0.0
    if int(warmup_epochs) <= 0:
        return 1.0
    progressed = int(epoch) - int(start_epoch) + 1
    return float(min(1.0, max(0.0, float(progressed) / float(max(1, int(warmup_epochs))))))


def _build_rank_examples(
    tracks: Tracks,
    interactions: list[Interaction],
) -> tuple[
    list[RankExample],
    dict[str, np.ndarray],
    dict[str, dict[str, np.ndarray]],
]:
    track_id_to_idx = {str(tid): int(i) for i, tid in enumerate(tracks.track_id.tolist())}
    all_idx = np.arange(len(tracks), dtype=np.int64)
    culture_values = tracks.culture.astype(str)
    culture_to_idx = {
        str(c): np.nonzero(culture_values == str(c))[0].astype(np.int64)
        for c in sorted({str(x) for x in culture_values.tolist()})
    }

    by_user: dict[str, list[Interaction]] = {}
    for it in interactions:
        if str(it.track_id) not in track_id_to_idx:
            continue
        by_user.setdefault(str(it.user_id), []).append(it)

    examples: list[RankExample] = []
    user_global_neg_pools: dict[str, np.ndarray] = {}
    user_culture_neg_pools: dict[str, dict[str, np.ndarray]] = {}
    for user_id, rows in by_user.items():
        pos_idx = np.array([track_id_to_idx[str(it.track_id)] for it in rows], dtype=np.int64)
        if int(pos_idx.size) <= 1:
            continue
        weights = np.array([float(it.weight) for it in rows], dtype=np.float64)
        weights = weights / max(1e-12, float(weights.sum()))
        seen = {int(i) for i in pos_idx.tolist()}
        user_global_neg_pools[str(user_id)] = np.array(
            [int(i) for i in all_idx.tolist() if int(i) not in seen],
            dtype=np.int64,
        )
        user_culture_neg_pools[str(user_id)] = {
            str(culture): np.array(
                [int(i) for i in pool.tolist() if int(i) not in seen],
                dtype=np.int64,
            )
            for culture, pool in culture_to_idx.items()
        }
        for j, pos in enumerate(pos_idx.tolist()):
            mask = np.ones_like(pos_idx, dtype=bool)
            mask[int(j)] = False
            context_idx = pos_idx[mask]
            context_w = weights[mask]
            if int(context_idx.size) <= 0:
                continue
            context_w = context_w / max(1e-12, float(context_w.sum()))
            examples.append(
                RankExample(
                    user_id=str(user_id),
                    pos_idx=int(pos),
                    pos_culture=str(culture_values[int(pos)]),
                    context_idx=tuple(int(x) for x in context_idx.tolist()),
                    context_w=tuple(float(x) for x in context_w.tolist()),
                    weight=float(weights[int(j)]),
                )
            )
    return examples, user_global_neg_pools, user_culture_neg_pools


def generate_toy(out_dir: str | Path, n_tracks: int = 3000, dim: int = 128, seed: int = 7) -> dict[str, str]:
    out_dir = generate_toy_data(out_dir=out_dir, n_tracks=n_tracks, dim=dim, seed=seed)
    return {
        "dir": str(out_dir),
        "tracks": str(out_dir / "tracks.npz"),
        "interactions": str(out_dir / "interactions.csv"),
        "meta": str(out_dir / "meta.txt"),
    }


def build_tracks_with_culturemert(
    metadata_csv: str | Path,
    out_tracks_path: str | Path,
    model_id: str = "ntua-slp/CultureMERT-95M",
    device: str | None = None,
    pooling: str = "mean",
    max_seconds: float | None = 30.0,
    limit: int | None = None,
    skip_errors: bool = False,
) -> dict[str, object]:
    return build_tracks_from_audio(
        metadata_csv=metadata_csv,
        out_npz=out_tracks_path,
        model_id=model_id,
        device=device,
        pooling=pooling,
        max_seconds=max_seconds,
        limit=limit,
        skip_errors=skip_errors,
    )


def train_model(
    tracks_path: str | Path,
    out_path: str | Path,
    constraints_path: str | Path | None = None,
    init_checkpoint_path: str | Path | None = None,
    strict_init: bool = True,
    epochs: int = 10,
    batch_size: int = 256,
    lr: float = 2e-3,
    seed: int = 42,
    prefer_cuda: bool = False,
    lambda_constraints: float = 0.1,
    constraint_margin: float = 1.0,
    lambda_domain: float = 0.5,
    lambda_contrast: float = 0.2,
    lambda_cov: float = 0.05,
    lambda_tc: float = 0.05,
    lambda_hsic: float = 0.02,
    beta_kl: float = 1.0,
    shared_encoder: bool = False,
    regularizer_warmup_epochs: int = 0,
    lambda_source: float = 0.0,
    source_balanced_batch: bool = False,
    interactions_path: str | Path | None = None,
    lambda_rank: float = 0.0,
    ranking_batch_size: int = 32,
    ranking_negatives: int = 4,
    ranking_margin: float = 0.2,
    ranking_same_culture_ratio: float = 0.5,
    constraint_batch_size: int = 64,
    constraint_candidate_pool_size: int = 256,
    constraint_hard_mining: bool = False,
    constraint_start_epoch: int = 0,
    constraint_warmup_epochs: int = 0,
    rank_start_epoch: int = 0,
    rank_warmup_epochs: int = 0,
) -> dict:
    set_seed(int(seed))
    device = get_device(bool(prefer_cuda))

    tracks = load_tracks(str(tracks_path))
    vocab = CultureVocab.from_tracks(tracks)
    source_vocab = SourceVocab.from_tracks(tracks) if tracks.source_dataset is not None else None
    ds = TrackDataset(tracks, vocab, source_vocab=source_vocab)
    if len(ds) == 0:
        raise RuntimeError("empty dataset: no tracks to train on")
    effective_batch_size = min(int(batch_size), len(ds))
    sampler = make_source_balanced_sampler(tracks) if bool(source_balanced_batch) else None
    dl = DataLoader(
        ds,
        batch_size=effective_batch_size,
        shuffle=bool(sampler is None),
        sampler=sampler,
        num_workers=0,
        collate_fn=collate_batch,
        drop_last=False,
    )

    lambda_affect = 0.0
    affect_classes = 8
    if tracks.affect_label is not None:
        lambda_affect = 0.2
        affect_classes = int(np.max(tracks.affect_label) + 1)

    cfg = DCASConfig(
        in_dim=tracks.dim,
        n_cultures=len(vocab.id_to_culture),
        n_sources=len(source_vocab.id_to_source) if source_vocab is not None else 0,
        lambda_affect=lambda_affect,
        affect_classes=affect_classes,
        lambda_domain=float(lambda_domain),
        lambda_contrast=float(lambda_contrast),
        lambda_cov=float(lambda_cov),
        lambda_tc=float(lambda_tc),
        lambda_hsic=float(lambda_hsic),
        lambda_source=float(lambda_source),
        beta_kl=float(beta_kl),
        shared_encoder=bool(shared_encoder),
    )
    model = DCASModel(cfg).to(device)
    warm_start_info: dict[str, object] | None = None
    if init_checkpoint_path is not None:
        init_model, init_vocab = load_checkpoint(str(init_checkpoint_path), map_location=str(device))
        if list(init_vocab.id_to_culture) != list(vocab.id_to_culture):
            raise ValueError("warm-start checkpoint culture vocabulary does not match the current tracks")
        load_result = model.load_state_dict(init_model.state_dict(), strict=bool(strict_init))
        warm_start_info = {
            "init_checkpoint_path": str(init_checkpoint_path),
            "strict_init": bool(strict_init),
            "missing_keys": list(getattr(load_result, "missing_keys", [])),
            "unexpected_keys": list(getattr(load_result, "unexpected_keys", [])),
        }
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr))

    constraints: list[PairwiseConstraint] | None = None
    if constraints_path is not None:
        constraints = load_constraints(str(constraints_path))
    track_id_to_idx = {str(tid): i for i, tid in enumerate(tracks.track_id.tolist())}
    x_all = torch.from_numpy(tracks.embedding.astype(np.float32)).to(device)

    rank_examples: list[RankExample] | None = None
    user_global_neg_pools: dict[str, np.ndarray] = {}
    user_culture_neg_pools: dict[str, dict[str, np.ndarray]] = {}
    if interactions_path is not None and float(lambda_rank) > 0:
        rank_interactions = load_interactions(str(interactions_path))
        rank_examples, user_global_neg_pools, user_culture_neg_pools = _build_rank_examples(
            tracks=tracks,
            interactions=rank_interactions,
        )

    def constraint_loss(sample: list[PairwiseConstraint], za_all: torch.Tensor) -> torch.Tensor:
        pairs = [c for c in sample if c.track_id_a in track_id_to_idx and c.track_id_b in track_id_to_idx]
        if not pairs:
            return torch.zeros((), device=device)
        idx_a = torch.tensor([track_id_to_idx[c.track_id_a] for c in pairs], device=device)
        idx_b = torch.tensor([track_id_to_idx[c.track_id_b] for c in pairs], device=device)
        similar = torch.tensor([1.0 if c.similar else 0.0 for c in pairs], device=device, dtype=torch.float32)
        za_a = za_all[idx_a]
        za_b = za_all[idx_b]
        dist = torch.norm(za_a - za_b, dim=-1)
        pos = (dist**2) * similar
        neg = (torch.relu(torch.tensor(float(constraint_margin), device=dist.device) - dist) ** 2) * (1.0 - similar)
        pair_loss = pos + neg
        if bool(constraint_hard_mining) and int(pair_loss.shape[0]) > int(constraint_batch_size):
            top_k = min(int(constraint_batch_size), int(pair_loss.shape[0]))
            top_idx = torch.topk(pair_loss.detach(), k=top_k, largest=True).indices
            pair_loss = pair_loss[top_idx]
        return pair_loss.mean()

    def ranking_loss(za_all: torch.Tensor) -> torch.Tensor:
        if not rank_examples:
            return torch.zeros((), device=device)
        sample = random.sample(rank_examples, k=min(int(ranking_batch_size), len(rank_examples)))
        if not sample:
            return torch.zeros((), device=device)
        margin_t = torch.tensor(float(ranking_margin), dtype=torch.float32, device=device)
        rank_losses: list[torch.Tensor] = []
        for ex in sample:
            context_idx = torch.tensor(ex.context_idx, dtype=torch.long, device=device)
            context_w = torch.tensor(ex.context_w, dtype=torch.float32, device=device)
            if int(context_idx.numel()) <= 0:
                continue
            user_vec = (za_all[context_idx] * context_w.unsqueeze(1)).sum(dim=0, keepdim=True)
            pos_vec = za_all[torch.tensor([int(ex.pos_idx)], dtype=torch.long, device=device)]
            pos_dist = torch.cdist(user_vec, pos_vec).squeeze()

            same_pool = user_culture_neg_pools.get(str(ex.user_id), {}).get(str(ex.pos_culture), np.array([], dtype=np.int64))
            global_pool = user_global_neg_pools.get(str(ex.user_id), np.array([], dtype=np.int64))
            neg_ids: list[int] = []
            for _ in range(max(1, int(ranking_negatives))):
                use_same = bool(same_pool.size > 0) and random.random() < float(ranking_same_culture_ratio)
                pool = same_pool if use_same else global_pool
                if int(pool.size) <= 0:
                    pool = global_pool if int(global_pool.size) > 0 else same_pool
                if int(pool.size) <= 0:
                    continue
                neg_ids.append(int(pool[random.randrange(int(pool.size))]))
            if not neg_ids:
                continue
            neg_vec = za_all[torch.tensor(neg_ids, dtype=torch.long, device=device)]
            neg_dist = torch.cdist(user_vec, neg_vec).squeeze(0)
            ex_loss = torch.relu(margin_t + pos_dist - neg_dist).mean()
            rank_losses.append(ex_loss * float(ex.weight))
        if not rank_losses:
            return torch.zeros((), device=device)
        return torch.stack(rank_losses).mean()

    history: list[dict[str, float]] = []
    for epoch in range(int(epochs)):
        model.train()
        losses: list[float] = []
        constraint_losses: list[float] = []
        rank_losses: list[float] = []
        warmup = int(regularizer_warmup_epochs)
        if warmup > 0:
            reg_scale = min(1.0, float(epoch + 1) / float(warmup))
        else:
            reg_scale = 1.0
        constraint_scale = _stage_scale(
            epoch=int(epoch),
            start_epoch=int(constraint_start_epoch),
            warmup_epochs=int(constraint_warmup_epochs),
        )
        rank_scale = _stage_scale(
            epoch=int(epoch),
            start_epoch=int(rank_start_epoch),
            warmup_epochs=int(rank_warmup_epochs),
        )
        for batch in dl:
            batch = type(batch)(
                x=batch.x.to(device),
                culture=batch.culture.to(device),
                track_index=batch.track_index.to(device),
                affect_label=batch.affect_label.to(device) if batch.affect_label is not None else None,
                source_label=batch.source_label.to(device) if batch.source_label is not None else None,
            )
            out = model(
                batch,
                reg_scales={
                    "domain": reg_scale,
                    "contrast": reg_scale,
                    "cov": reg_scale,
                    "tc": reg_scale,
                    "hsic": reg_scale,
                    "affect": reg_scale,
                    "source": reg_scale,
                },
            )
            loss = out["loss"]
            aux_za_all: torch.Tensor | None = None
            need_aux = (
                constraints is not None
                and float(lambda_constraints) > 0
                and float(constraint_scale) > 0
            ) or (
                rank_examples is not None
                and float(lambda_rank) > 0
                and float(rank_scale) > 0
            )
            if need_aux:
                _, _, aux_za_all = model.encode(x_all)
            if constraints is not None and float(lambda_constraints) > 0 and float(constraint_scale) > 0 and aux_za_all is not None:
                sample_k = min(
                    int(len(constraints)),
                    int(max(constraint_batch_size, constraint_candidate_pool_size))
                    if bool(constraint_hard_mining)
                    else int(constraint_batch_size),
                )
                sample = random.sample(constraints, k=max(1, sample_k))
                c_loss = constraint_loss(sample, aux_za_all)
                loss = loss + float(lambda_constraints) * float(constraint_scale) * c_loss
                constraint_losses.append(float(c_loss.detach().cpu().item()))
            if rank_examples is not None and float(lambda_rank) > 0 and float(rank_scale) > 0 and aux_za_all is not None:
                r_loss = ranking_loss(aux_za_all)
                loss = loss + float(lambda_rank) * float(rank_scale) * r_loss
                rank_losses.append(float(r_loss.detach().cpu().item()))

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu().item()))

        if not losses:
            raise RuntimeError("no training batches were produced; check dataset size and batch_size")
        history.append(
            {
                "epoch": float(epoch),
                "loss": float(np.mean(losses)) if losses else float("nan"),
                "constraint_loss": float(np.mean(constraint_losses)) if constraint_losses else 0.0,
                "rank_loss": float(np.mean(rank_losses)) if rank_losses else 0.0,
                "regularizer_scale": float(reg_scale),
                "constraint_scale": float(constraint_scale),
                "rank_scale": float(rank_scale),
            }
        )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(str(out_path), model, vocab)
    return {
        "checkpoint": str(out_path),
        "history": history,
        "cfg": asdict(cfg),
        "cultures": vocab.id_to_culture,
        "n_constraints": int(len(constraints)) if constraints is not None else 0,
        "n_rank_examples": int(len(rank_examples)) if rank_examples is not None else 0,
        "warm_start": warm_start_info,
    }


def recommend(
    model_path: str | Path,
    tracks_path: str | Path,
    interactions_path: str | Path,
    user_id: str,
    target_culture: str,
    k: int = 20,
    prefer_cuda: bool = False,
    epsilon: float = 0.1,
    iters: int = 200,
) -> dict:
    device = torch.device("cuda" if prefer_cuda and torch.cuda.is_available() else "cpu")
    model, _ = load_checkpoint(str(model_path), map_location=str(device))
    tracks = load_tracks(str(tracks_path))
    interactions: list[Interaction] = load_interactions(str(interactions_path))
    recs, metrics = recommend_ot(
        model=model,
        tracks=tracks,
        interactions=interactions,
        user_id=user_id,
        target_culture=target_culture,
        k=int(k),
        device=device,
        epsilon=float(epsilon),
        iters=int(iters),
    )
    return {
        "metrics": metrics,
        "recommendations": [asdict(r) for r in recs],
    }


def style_transfer(
    model_path: str | Path,
    tracks_path: str | Path,
    source_track_id: str,
    style_track_id: str,
    out_path: str | Path,
    target_culture: str | None = None,
    alpha: float = 1.0,
    k: int = 10,
    prefer_cuda: bool = False,
) -> dict:
    device = torch.device("cuda" if prefer_cuda and torch.cuda.is_available() else "cpu")
    model, _ = load_checkpoint(str(model_path), map_location=str(device))
    tracks = load_tracks(str(tracks_path))
    emb, neighbors, meta = generate_counterfactual_embedding(
        model=model,
        tracks=tracks,
        source_track_id=source_track_id,
        style_track_id=style_track_id,
        target_culture=target_culture,
        alpha=float(alpha),
        k=int(k),
        device=device,
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(out_path),
        generated_embedding=emb,
        source_track_id=np.array([source_track_id], dtype="<U128"),
        style_track_id=np.array([style_track_id], dtype="<U128"),
        target_culture=np.array([target_culture or ""], dtype="<U128"),
        alpha=np.array([float(alpha)], dtype=np.float32),
    )

    return {
        "artifact": str(out_path),
        "neighbors": [asdict(n) for n in neighbors],
        "meta": meta,
        "dim": int(emb.shape[0]),
    }


def style_transfer_waveform(
    source_audio_path: str | Path,
    style_audio_path: str | Path,
    out_wav_path: str | Path,
    alpha: float = 0.7,
    target_sr: int = 24000,
    n_fft: int = 1024,
    hop_length: int = 256,
    win_length: int = 1024,
    max_seconds: float | None = 12.0,
    peak_norm: float = 0.98,
) -> dict:
    out = transfer_waveform_style(
        source_audio_path=source_audio_path,
        style_audio_path=style_audio_path,
        output_wav_path=out_wav_path,
        alpha=float(alpha),
        target_sr=int(target_sr),
        n_fft=int(n_fft),
        hop_length=int(hop_length),
        win_length=int(win_length),
        max_seconds=float(max_seconds) if max_seconds is not None else None,
        peak_norm=float(peak_norm),
    )
    return {
        "artifact": out.output_path,
        "sample_rate": int(out.sample_rate),
        "n_samples": int(out.n_samples),
        "source_audio_path": out.source_audio_path,
        "style_audio_path": out.style_audio_path,
        "metrics": out.metrics,
    }


def pal_tasks(
    model_path: str | Path,
    tracks_path: str | Path,
    out_path: str | Path,
    n: int = 100,
    prefer_cuda: bool = False,
    uncertainty_method: str = "auto",
) -> dict:
    device = torch.device("cuda" if prefer_cuda and torch.cuda.is_available() else "cpu")
    model, _ = load_checkpoint(str(model_path), map_location=str(device))
    tracks: Tracks = load_tracks(str(tracks_path))
    ranked = rank_by_uncertainty(model=model, tracks=tracks, device=device, method=str(uncertainty_method))
    top = ranked[: int(n)]

    track_id_to_idx = {str(tid): i for i, tid in enumerate(tracks.track_id.tolist())}
    x_all = torch.from_numpy(tracks.embedding.astype(np.float32)).to(device)
    model.eval()
    model.to(device)
    with torch.no_grad():
        _, _, za_mu = model.encode(x_all)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for tid, score in top:
            idx = track_id_to_idx[tid]
            z = za_mu[idx : idx + 1]
            d = torch.cdist(z, za_mu).squeeze(0)
            nn = int(torch.topk(d, k=6, largest=False).indices[1].item())
            obj = {
                "track_id": tid,
                "culture": str(tracks.culture[idx]),
                "uncertainty": float(score),
                "uncertainty_method": str(uncertainty_method),
                "compare_to": str(tracks.track_id[nn]),
                "question": PAL_TASK_QUESTION_ZH,
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    return {"tasks": str(out_path), "count": int(len(top)), "uncertainty_method": str(uncertainty_method)}
