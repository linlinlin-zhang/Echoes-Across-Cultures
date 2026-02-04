from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def kl_standard_normal(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return 0.5 * torch.sum(torch.exp(logvar) + mu**2 - 1.0 - logvar, dim=-1).mean()


def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


def cov_offdiag_loss(z: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    zc = z - z.mean(dim=0, keepdim=True)
    cov = (zc.T @ zc) / (zc.shape[0] - 1 + eps)
    off = cov - torch.diag(torch.diag(cov))
    return (off**2).mean()


def info_nce(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    logits = (z1 @ z2.T) / max(1e-6, float(temperature))
    labels = torch.arange(z1.shape[0], device=z1.device)
    return F.cross_entropy(logits, labels)


def cosine_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    return 1.0 - torch.sum(a * b, dim=-1)


def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    p = F.softmax(logits, dim=-1)
    logp = torch.log(p.clamp_min(1e-12))
    return -(p * logp).sum(dim=-1)


def schedule_grl_scale(step: int, max_step: int) -> float:
    if max_step <= 0:
        return 1.0
    p = min(1.0, max(0.0, step / max_step))
    return float(2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0)

