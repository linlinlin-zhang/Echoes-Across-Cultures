from __future__ import annotations

import argparse
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from dcas.data.batch import collate_batch
from dcas.data.npz_tracks import load_tracks
from dcas.data.torch_dataset import CultureVocab, TrackDataset
from dcas.models.dcas_vae import DCASConfig, DCASModel
from dcas.pal.constraints import load_constraints
from dcas.serialization import save_checkpoint
from dcas.utils import get_device, set_seed


def _constraint_loss(
    model: DCASModel,
    emb_a: torch.Tensor,
    emb_b: torch.Tensor,
    similar: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    _, _, za_a = model.encode(emb_a)
    _, _, za_b = model.encode(emb_b)
    dist = torch.norm(za_a - za_b, dim=-1)
    pos = (dist**2) * similar
    neg = (torch.relu(torch.tensor(float(margin), device=dist.device) - dist) ** 2) * (1.0 - similar)
    return (pos + neg).mean()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="tracks.npz")
    ap.add_argument("--out", required=True, help="model checkpoint path")
    ap.add_argument("--constraints", default=None, help="pairwise constraints jsonl")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--prefer_cuda", action="store_true")
    ap.add_argument("--lambda_constraints", type=float, default=0.1)
    ap.add_argument("--constraint_margin", type=float, default=1.0)
    ap.add_argument("--lambda_domain", type=float, default=0.5)
    ap.add_argument("--lambda_contrast", type=float, default=0.2)
    ap.add_argument("--lambda_cov", type=float, default=0.05)
    ap.add_argument("--lambda_tc", type=float, default=0.05)
    ap.add_argument("--lambda_hsic", type=float, default=0.02)
    ap.add_argument("--beta_kl", type=float, default=1.0)
    ap.add_argument("--shared_encoder", action="store_true")
    ap.add_argument("--regularizer_warmup_epochs", type=int, default=0)
    args = ap.parse_args()

    set_seed(int(args.seed))
    device = get_device(bool(args.prefer_cuda))

    tracks = load_tracks(args.data)
    vocab = CultureVocab.from_tracks(tracks)
    ds = TrackDataset(tracks, vocab)
    if len(ds) == 0:
        raise RuntimeError("empty dataset: no tracks to train on")
    effective_batch_size = min(int(args.batch_size), len(ds))
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
        lambda_domain=float(args.lambda_domain),
        lambda_contrast=float(args.lambda_contrast),
        lambda_cov=float(args.lambda_cov),
        lambda_tc=float(args.lambda_tc),
        lambda_hsic=float(args.lambda_hsic),
        beta_kl=float(args.beta_kl),
        shared_encoder=bool(args.shared_encoder),
    )
    model = DCASModel(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr))

    constraints = None
    track_id_to_idx = {str(tid): i for i, tid in enumerate(tracks.track_id.tolist())}
    if args.constraints:
        constraints = load_constraints(args.constraints)

    x_all = torch.from_numpy(tracks.embedding.astype(np.float32)).to(device)

    for epoch in range(int(args.epochs)):
        model.train()
        losses: list[float] = []
        warmup = int(args.regularizer_warmup_epochs)
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

            if constraints and float(args.lambda_constraints) > 0:
                sample = random.sample(constraints, k=min(64, len(constraints)))
                idx_a = [track_id_to_idx[c.track_id_a] for c in sample if c.track_id_a in track_id_to_idx and c.track_id_b in track_id_to_idx]
                idx_b = [track_id_to_idx[c.track_id_b] for c in sample if c.track_id_a in track_id_to_idx and c.track_id_b in track_id_to_idx]
                sim = [1.0 if c.similar else 0.0 for c in sample if c.track_id_a in track_id_to_idx and c.track_id_b in track_id_to_idx]
                if idx_a:
                    emb_a = x_all[torch.tensor(idx_a, device=device)]
                    emb_b = x_all[torch.tensor(idx_b, device=device)]
                    similar = torch.tensor(sim, device=device, dtype=torch.float32)
                    c_loss = _constraint_loss(model, emb_a, emb_b, similar, margin=float(args.constraint_margin))
                    loss = loss + float(args.lambda_constraints) * c_loss

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            losses.append(float(loss.detach().cpu().item()))

        if not losses:
            raise RuntimeError("no training batches were produced; check dataset size and batch_size")
        avg = float(np.mean(losses)) if losses else float("nan")
        print(f"epoch={epoch} loss={avg:.4f}")

    save_checkpoint(args.out, model, vocab)
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()

