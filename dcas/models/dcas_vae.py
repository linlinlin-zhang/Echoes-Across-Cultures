from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from dcas.utils import Batch

from .layers import GradientReversal
from .losses import (
    cov_offdiag_loss,
    gaussian_total_correlation,
    hsic_rbf,
    info_nce,
    kl_standard_normal,
    reparameterize,
)
from .mlp import GaussianHead, MLPConfig, make_mlp


@dataclass(frozen=True)
class DCASConfig:
    in_dim: int
    n_cultures: int
    zc_dim: int = 32
    zs_dim: int = 32
    za_dim: int = 16
    hidden_dim: int = 256
    depth: int = 3
    dropout: float = 0.1
    beta_kl: float = 1.0
    lambda_domain: float = 0.5
    lambda_contrast: float = 0.2
    lambda_cov: float = 0.05
    lambda_tc: float = 0.05
    lambda_hsic: float = 0.02
    lambda_affect: float = 0.0
    affect_classes: int = 8
    grl_scale: float = 1.0
    contrast_temperature: float = 0.2
    aug_noise_std: float = 0.02


class DCASModel(nn.Module):
    def __init__(self, cfg: DCASConfig):
        super().__init__()
        self.cfg = cfg

        self.encoder = make_mlp(
            MLPConfig(
                in_dim=cfg.in_dim,
                hidden_dim=cfg.hidden_dim,
                out_dim=cfg.hidden_dim,
                depth=cfg.depth,
                dropout=cfg.dropout,
            )
        )
        self.zc_head = GaussianHead(cfg.hidden_dim, cfg.zc_dim)
        self.zs_head = GaussianHead(cfg.hidden_dim, cfg.zs_dim)
        self.za_head = GaussianHead(cfg.hidden_dim, cfg.za_dim)

        self.decoder = make_mlp(
            MLPConfig(
                in_dim=cfg.zc_dim + cfg.zs_dim + cfg.za_dim,
                hidden_dim=cfg.hidden_dim,
                out_dim=cfg.in_dim,
                depth=cfg.depth,
                dropout=cfg.dropout,
            )
        )

        self.grl = GradientReversal(scale=cfg.grl_scale)
        self.culture_disc = make_mlp(
            MLPConfig(
                in_dim=cfg.za_dim,
                hidden_dim=cfg.hidden_dim,
                out_dim=cfg.n_cultures,
                depth=2,
                dropout=cfg.dropout,
            )
        )
        self.affect_head = make_mlp(
            MLPConfig(
                in_dim=cfg.za_dim,
                hidden_dim=cfg.hidden_dim,
                out_dim=cfg.affect_classes,
                depth=2,
                dropout=cfg.dropout,
            )
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        zc_mu, _ = self.zc_head(h)
        zs_mu, _ = self.zs_head(h)
        za_mu, _ = self.za_head(h)
        return zc_mu, zs_mu, za_mu

    def forward(self, batch: Batch, reg_scales: dict[str, float] | None = None) -> dict[str, torch.Tensor]:
        x = batch.x
        h = self.encoder(x)
        scales = reg_scales or {}
        s_domain = float(scales.get("domain", 1.0))
        s_contrast = float(scales.get("contrast", 1.0))
        s_cov = float(scales.get("cov", 1.0))
        s_tc = float(scales.get("tc", 1.0))
        s_hsic = float(scales.get("hsic", 1.0))
        s_affect = float(scales.get("affect", 1.0))

        zc_mu, zc_logvar = self.zc_head(h)
        zs_mu, zs_logvar = self.zs_head(h)
        za_mu, za_logvar = self.za_head(h)

        zc = reparameterize(zc_mu, zc_logvar)
        zs = reparameterize(zs_mu, zs_logvar)
        za = reparameterize(za_mu, za_logvar)

        z = torch.cat([zc, zs, za], dim=-1)
        x_hat = self.decoder(z)

        recon = F.mse_loss(x_hat, x, reduction="mean")
        kl = kl_standard_normal(zc_mu, zc_logvar) + kl_standard_normal(zs_mu, zs_logvar) + kl_standard_normal(za_mu, za_logvar)

        domain_logits = self.culture_disc(self.grl(za))
        domain = F.cross_entropy(domain_logits, batch.culture)

        x_aug = x + torch.randn_like(x) * float(self.cfg.aug_noise_std)
        h_aug = self.encoder(x_aug)
        zc_mu_aug, _ = self.zc_head(h_aug)
        contrast = info_nce(zc_mu, zc_mu_aug, temperature=self.cfg.contrast_temperature)

        cov = cov_offdiag_loss(torch.cat([zc_mu, zs_mu, za_mu], dim=-1))
        tc = gaussian_total_correlation(torch.cat([zc_mu, zs_mu, za_mu], dim=-1))
        hsic = hsic_rbf(zc_mu, zs_mu) + hsic_rbf(zc_mu, za_mu) + hsic_rbf(zs_mu, za_mu)

        affect = torch.zeros((), device=x.device)
        affect_acc = torch.zeros((), device=x.device)
        if batch.affect_label is not None and self.cfg.lambda_affect > 0:
            affect_logits = self.affect_head(za)
            affect = F.cross_entropy(affect_logits, batch.affect_label)
            affect_acc = (affect_logits.argmax(dim=-1) == batch.affect_label).float().mean()

        loss = (
            recon
            + self.cfg.beta_kl * kl
            + self.cfg.lambda_domain * s_domain * domain
            + self.cfg.lambda_contrast * s_contrast * contrast
            + self.cfg.lambda_cov * s_cov * cov
            + self.cfg.lambda_tc * s_tc * tc
            + self.cfg.lambda_hsic * s_hsic * hsic
            + self.cfg.lambda_affect * s_affect * affect
        )

        return {
            "loss": loss,
            "recon": recon.detach(),
            "kl": kl.detach(),
            "domain": domain.detach(),
            "contrast": contrast.detach(),
            "cov": cov.detach(),
            "tc": tc.detach(),
            "hsic": hsic.detach(),
            "affect": affect.detach(),
            "affect_acc": affect_acc.detach(),
        }

