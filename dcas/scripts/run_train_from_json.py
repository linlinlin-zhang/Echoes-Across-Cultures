from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dcas.pipelines import train_model


def main() -> None:
    ap = argparse.ArgumentParser(description="Run DCAS training from a JSON config file.")
    ap.add_argument("--config", required=True, help="Path to JSON config")
    args = ap.parse_args()

    config_path = Path(args.config)
    with open(config_path, "r", encoding="utf-8-sig") as f:
        cfg = json.load(f)

    out = train_model(
        tracks_path=str(cfg["tracks"]),
        out_path=str(cfg["out"]),
        constraints_path=str(cfg["constraints"]) if cfg.get("constraints") else None,
        epochs=int(cfg.get("epochs", 10)),
        batch_size=int(cfg.get("batch_size", 256)),
        lr=float(cfg.get("lr", 2e-3)),
        seed=int(cfg.get("seed", 42)),
        prefer_cuda=bool(cfg.get("prefer_cuda", False)),
        lambda_constraints=float(cfg.get("lambda_constraints", 0.1)),
        constraint_margin=float(cfg.get("constraint_margin", 1.0)),
        lambda_domain=float(cfg.get("lambda_domain", 0.5)),
        lambda_contrast=float(cfg.get("lambda_contrast", 0.2)),
        lambda_cov=float(cfg.get("lambda_cov", 0.05)),
        lambda_tc=float(cfg.get("lambda_tc", 0.05)),
        lambda_hsic=float(cfg.get("lambda_hsic", 0.02)),
        beta_kl=float(cfg.get("beta_kl", 1.0)),
        shared_encoder=bool(cfg.get("shared_encoder", False)),
        regularizer_warmup_epochs=int(cfg.get("regularizer_warmup_epochs", 0)),
        lambda_source=float(cfg.get("lambda_source", 0.0)),
        source_balanced_batch=bool(cfg.get("source_balanced_batch", False)),
    )
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
