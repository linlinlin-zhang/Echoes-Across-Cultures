from __future__ import annotations

import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dcas.data.batch import collate_batch
from dcas.data.interactions import Interaction, load_interactions
from dcas.data.npz_tracks import Tracks, load_tracks
from dcas.data.torch_dataset import CultureVocab, TrackDataset
from dcas.models.dcas_vae import DCASConfig, DCASModel
from dcas.pal.constraints import PairwiseConstraint, load_constraints
from dcas.pal.uncertainty import rank_by_uncertainty
from dcas.recommender import Recommendation, recommend_ot
from dcas.scripts.build_tracks_from_audio import build_tracks_from_audio
from dcas.scripts.make_toy_data import generate_toy_data
from dcas.serialization import load_checkpoint, save_checkpoint
from dcas.style_transfer import generate_counterfactual_embedding
from dcas.utils import get_device, set_seed
from dcas.waveform_style_transfer import transfer_waveform_style


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
    regularizer_warmup_epochs: int = 0,
) -> dict:
    set_seed(int(seed))
    device = get_device(bool(prefer_cuda))

    tracks = load_tracks(str(tracks_path))
    vocab = CultureVocab.from_tracks(tracks)
    ds = TrackDataset(tracks, vocab)
    if len(ds) == 0:
        raise RuntimeError("empty dataset: no tracks to train on")
    effective_batch_size = min(int(batch_size), len(ds))
    dl = DataLoader(
        ds,
        batch_size=effective_batch_size,
        shuffle=True,
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
        lambda_affect=lambda_affect,
        affect_classes=affect_classes,
        lambda_domain=float(lambda_domain),
        lambda_contrast=float(lambda_contrast),
        lambda_cov=float(lambda_cov),
        lambda_tc=float(lambda_tc),
        lambda_hsic=float(lambda_hsic),
    )
    model = DCASModel(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr))

    constraints: list[PairwiseConstraint] | None = None
    if constraints_path is not None:
        constraints = load_constraints(str(constraints_path))
    track_id_to_idx = {str(tid): i for i, tid in enumerate(tracks.track_id.tolist())}
    x_all = torch.from_numpy(tracks.embedding.astype(np.float32)).to(device)

    def constraint_loss(sample: list[PairwiseConstraint]) -> torch.Tensor:
        pairs = [c for c in sample if c.track_id_a in track_id_to_idx and c.track_id_b in track_id_to_idx]
        if not pairs:
            return torch.zeros((), device=device)
        idx_a = torch.tensor([track_id_to_idx[c.track_id_a] for c in pairs], device=device)
        idx_b = torch.tensor([track_id_to_idx[c.track_id_b] for c in pairs], device=device)
        similar = torch.tensor([1.0 if c.similar else 0.0 for c in pairs], device=device, dtype=torch.float32)
        emb_a = x_all[idx_a]
        emb_b = x_all[idx_b]
        _, _, za_a = model.encode(emb_a)
        _, _, za_b = model.encode(emb_b)
        dist = torch.norm(za_a - za_b, dim=-1)
        pos = (dist**2) * similar
        neg = (torch.relu(torch.tensor(float(constraint_margin), device=dist.device) - dist) ** 2) * (1.0 - similar)
        return (pos + neg).mean()

    history: list[dict[str, float]] = []
    for epoch in range(int(epochs)):
        model.train()
        losses: list[float] = []
        warmup = int(regularizer_warmup_epochs)
        if warmup > 0:
            reg_scale = min(1.0, float(epoch + 1) / float(warmup))
        else:
            reg_scale = 1.0
        for batch in dl:
            batch = type(batch)(
                x=batch.x.to(device),
                culture=batch.culture.to(device),
                track_index=batch.track_index.to(device),
                affect_label=batch.affect_label.to(device) if batch.affect_label is not None else None,
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
                },
            )
            loss = out["loss"]
            if constraints is not None and float(lambda_constraints) > 0:
                sample = random.sample(constraints, k=min(64, len(constraints)))
                loss = loss + float(lambda_constraints) * constraint_loss(sample)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu().item()))

        if not losses:
            raise RuntimeError("no training batches were produced; check dataset size and batch_size")
        history.append({"epoch": float(epoch), "loss": float(np.mean(losses)) if losses else float("nan")})

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(str(out_path), model, vocab)
    return {"checkpoint": str(out_path), "history": history, "cfg": asdict(cfg), "cultures": vocab.id_to_culture}


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
) -> dict:
    device = torch.device("cuda" if prefer_cuda and torch.cuda.is_available() else "cpu")
    model, _ = load_checkpoint(str(model_path), map_location=str(device))
    tracks: Tracks = load_tracks(str(tracks_path))
    ranked = rank_by_uncertainty(model=model, tracks=tracks, device=device)
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
                "compare_to": str(tracks.track_id[nn]),
                "question": "它们在情感/功能上是否相似？如果相似/不相似，请给出理由（rationale）。",
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    return {"tasks": str(out_path), "count": int(len(top))}
