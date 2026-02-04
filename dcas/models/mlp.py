from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class MLPConfig:
    in_dim: int
    hidden_dim: int
    out_dim: int
    depth: int = 2
    dropout: float = 0.1


def make_mlp(cfg: MLPConfig) -> nn.Module:
    layers: list[nn.Module] = []
    d = cfg.in_dim
    for _ in range(max(0, cfg.depth - 1)):
        layers.append(nn.Linear(d, cfg.hidden_dim))
        layers.append(nn.GELU())
        if cfg.dropout > 0:
            layers.append(nn.Dropout(cfg.dropout))
        d = cfg.hidden_dim
    layers.append(nn.Linear(d, cfg.out_dim))
    return nn.Sequential(*layers)


class GaussianHead(nn.Module):
    def __init__(self, in_dim: int, z_dim: int):
        super().__init__()
        self.mu = nn.Linear(in_dim, z_dim)
        self.logvar = nn.Linear(in_dim, z_dim)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.mu(h), self.logvar(h)

