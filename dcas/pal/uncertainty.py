from __future__ import annotations

import numpy as np
import torch

from dcas.data.npz_tracks import Tracks
from dcas.models.dcas_vae import DCASModel
from dcas.models.losses import entropy_from_logits


def rank_by_uncertainty(
    model: DCASModel,
    tracks: Tracks,
    device: torch.device | None = None,
    batch_size: int = 512,
) -> list[tuple[str, float]]:
    if device is None:
        device = torch.device("cpu")
    model.eval()
    model.to(device)

    x = torch.from_numpy(tracks.embedding.astype(np.float32))
    scores: list[tuple[str, float]] = []

    with torch.no_grad():
        for start in range(0, x.shape[0], int(batch_size)):
            xb = x[start : start + int(batch_size)].to(device)
            _, _, za_mu = model.encode(xb)
            logits = model.affect_head(za_mu)
            ent = entropy_from_logits(logits).detach().cpu().numpy()
            for i, s in enumerate(ent.tolist()):
                tid = str(tracks.track_id[start + i])
                scores.append((tid, float(s)))

    scores.sort(key=lambda t: -t[1])
    return scores

