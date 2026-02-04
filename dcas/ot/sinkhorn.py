from __future__ import annotations

import torch


def squared_euclidean_cost(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x2 = (x**2).sum(dim=-1, keepdim=True)
    y2 = (y**2).sum(dim=-1, keepdim=True).T
    xy = x @ y.T
    return (x2 + y2 - 2.0 * xy).clamp_min(0.0)


def sinkhorn_plan(
    a: torch.Tensor,
    b: torch.Tensor,
    cost: torch.Tensor,
    epsilon: float = 0.1,
    iters: int = 100,
    tol: float | None = 1e-6,
) -> torch.Tensor:
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("a and b must be 1D")
    if cost.ndim != 2:
        raise ValueError("cost must be 2D")
    if cost.shape[0] != a.shape[0] or cost.shape[1] != b.shape[0]:
        raise ValueError("cost shape must match a and b")

    eps = float(epsilon)
    log_a = torch.log(a.clamp_min(1e-12))
    log_b = torch.log(b.clamp_min(1e-12))
    log_K = -cost / max(1e-12, eps)

    u = torch.zeros_like(log_a)
    v = torch.zeros_like(log_b)

    for _ in range(int(iters)):
        u_prev = u
        u = log_a - torch.logsumexp(log_K + v[None, :], dim=1)
        v = log_b - torch.logsumexp(log_K + u[:, None], dim=0)
        if tol is not None:
            if torch.max(torch.abs(u - u_prev)).item() < float(tol):
                break

    log_P = log_K + u[:, None] + v[None, :]
    return torch.exp(log_P)

