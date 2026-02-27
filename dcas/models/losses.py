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


def gaussian_total_correlation(z: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Gaussian approximation of total correlation:
      TC(z) ~= 0.5 * (sum(log var_i) - logdet(Cov(z))).
    Returns a non-negative scalar (up to numerical error).
    """
    if z.ndim != 2:
        raise ValueError("z must be 2D [batch, dim]")
    n, d = z.shape
    if n <= 1 or d <= 1:
        return torch.zeros((), device=z.device, dtype=z.dtype)

    zc = z - z.mean(dim=0, keepdim=True)
    cov = (zc.T @ zc) / max(1, n - 1)
    eye = torch.eye(d, device=z.device, dtype=z.dtype)
    cov = cov + float(eps) * eye

    var = torch.diag(cov).clamp_min(float(eps))
    sum_log_var = torch.log(var).sum()

    sign, logdet = torch.linalg.slogdet(cov)
    if bool((sign <= 0).item()):
        cov = cov + float(eps) * eye
        sign, logdet = torch.linalg.slogdet(cov)
    if bool((sign <= 0).item()):
        return torch.zeros((), device=z.device, dtype=z.dtype)

    tc = 0.5 * (sum_log_var - logdet)
    return torch.clamp(tc, min=0.0)


def _rbf_kernel(x: torch.Tensor, sigma: float | None = None, eps: float = 1e-8) -> torch.Tensor:
    if x.ndim != 2:
        raise ValueError("x must be 2D [batch, dim]")
    dist2 = torch.cdist(x, x, p=2.0) ** 2
    if sigma is None or sigma <= 0:
        # Median heuristic from off-diagonal distances.
        m = dist2.shape[0]
        if m <= 1:
            sigma_val = 1.0
        else:
            tri = torch.triu_indices(m, m, offset=1, device=x.device)
            off = dist2[tri[0], tri[1]]
            med = torch.median(off) if off.numel() > 0 else torch.tensor(1.0, device=x.device, dtype=x.dtype)
            sigma_val = float(torch.sqrt(torch.clamp(med, min=eps)).item())
    else:
        sigma_val = float(sigma)
    gamma = 1.0 / max(eps, 2.0 * sigma_val * sigma_val)
    return torch.exp(-gamma * dist2)


def hsic_rbf(
    x: torch.Tensor,
    y: torch.Tensor,
    sigma_x: float | None = None,
    sigma_y: float | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Biased HSIC estimator with RBF kernels.
    Smaller value => weaker statistical dependence.
    """
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("x and y must be 2D [batch, dim]")
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must have same batch size")
    n = x.shape[0]
    if n <= 2:
        return torch.zeros((), device=x.device, dtype=x.dtype)

    k = _rbf_kernel(x, sigma=sigma_x, eps=eps)
    l = _rbf_kernel(y, sigma=sigma_y, eps=eps)
    h = torch.eye(n, device=x.device, dtype=x.dtype) - (1.0 / float(n)) * torch.ones((n, n), device=x.device, dtype=x.dtype)
    kh = h @ k @ h
    lh = h @ l @ h
    hsic = torch.trace(kh @ lh) / ((n - 1) ** 2)
    return torch.clamp(hsic, min=0.0)


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

