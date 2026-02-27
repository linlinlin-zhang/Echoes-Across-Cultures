from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from dcas.data.npz_tracks import Tracks
from dcas.models.dcas_vae import DCASModel


@dataclass(frozen=True)
class StyleTransferNeighbor:
    track_id: str
    culture: str
    distance: float


def generate_counterfactual_embedding(
    model: DCASModel,
    tracks: Tracks,
    source_track_id: str,
    style_track_id: str,
    target_culture: str | None = None,
    alpha: float = 1.0,
    k: int = 10,
    device: torch.device | None = None,
) -> tuple[np.ndarray, list[StyleTransferNeighbor], dict[str, float]]:
    if device is None:
        device = torch.device("cpu")
    model.eval()
    model.to(device)

    track_id_to_idx = {str(tid): i for i, tid in enumerate(tracks.track_id.tolist())}
    if source_track_id not in track_id_to_idx:
        raise ValueError(f"source track not found: {source_track_id}")
    if style_track_id not in track_id_to_idx:
        raise ValueError(f"style track not found: {style_track_id}")

    src_idx = int(track_id_to_idx[source_track_id])
    sty_idx = int(track_id_to_idx[style_track_id])
    x_all = torch.from_numpy(tracks.embedding.astype(np.float32)).to(device)

    with torch.no_grad():
        zc_all, zs_all, za_all = model.encode(x_all)
        zc_src = zc_all[src_idx : src_idx + 1]
        zs_src = zs_all[src_idx : src_idx + 1]
        zs_sty = zs_all[sty_idx : sty_idx + 1]
        za_src = za_all[src_idx : src_idx + 1]

        a = float(alpha)
        zs_new = (1.0 - a) * zs_src + a * zs_sty
        z_new = torch.cat([zc_src, zs_new, za_src], dim=-1)
        x_cf = model.decoder(z_new).squeeze(0)

    if target_culture is None:
        cand_idx = np.arange(len(tracks), dtype=np.int64)
    else:
        cand_idx = tracks.indices_of_cultures([target_culture])
    cand_idx = np.array([i for i in cand_idx.tolist() if i != src_idx], dtype=np.int64)
    if cand_idx.size == 0:
        raise ValueError("no candidates found for target pool")

    cand_x = x_all[torch.from_numpy(cand_idx).to(device)]
    with torch.no_grad():
        dist = torch.cdist(x_cf.unsqueeze(0), cand_x).squeeze(0).detach().cpu().numpy()

    order = np.argsort(dist)[: int(k)]
    neighbors: list[StyleTransferNeighbor] = []
    for o in order.tolist():
        idx = int(cand_idx[o])
        neighbors.append(
            StyleTransferNeighbor(
                track_id=str(tracks.track_id[idx]),
                culture=str(tracks.culture[idx]),
                distance=float(dist[o]),
            )
        )

    with torch.no_grad():
        d_zs_shift = torch.norm((zs_new - zs_src), dim=-1).mean().item()
        d_zc_drift = torch.norm((zc_src - zc_all[src_idx : src_idx + 1]), dim=-1).mean().item()
        d_za_drift = torch.norm((za_src - za_all[src_idx : src_idx + 1]), dim=-1).mean().item()

    meta = {
        "zs_shift": float(d_zs_shift),
        "zc_drift": float(d_zc_drift),
        "za_drift": float(d_za_drift),
    }
    return x_cf.detach().cpu().numpy().astype(np.float32), neighbors, meta

