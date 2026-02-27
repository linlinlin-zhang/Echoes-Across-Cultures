from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dcas.pipelines import train_model
from dcas.scripts.evaluate_recommender import evaluate_recommender


def _row(name: str, ev: dict[str, Any]) -> dict[str, Any]:
    s = ev["summary"]
    return {
        "name": name,
        "n_user_culture_evals": int(s["n_user_culture_evals"]),
        "serendipity_mean": float(s["serendipity_mean"]),
        "cultural_calibration_kl_mean": float(s["cultural_calibration_kl_mean"]),
    }


def _write_markdown(path: str | Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    base = rows[0]
    lines = [
        "# Ablation Draft Table",
        "",
        "| setting | serendipity_mean | delta_vs_full | cultural_calibration_kl_mean | delta_vs_full | evals |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        d_ser = float(r["serendipity_mean"]) - float(base["serendipity_mean"])
        d_kl = float(r["cultural_calibration_kl_mean"]) - float(base["cultural_calibration_kl_mean"])
        lines.append(
            f"| {r['name']} | {r['serendipity_mean']:.10f} | {d_ser:+.10f} | "
            f"{r['cultural_calibration_kl_mean']:.10f} | {d_kl:+.10f} | {int(r['n_user_culture_evals'])} |"
        )
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_ablation(
    tracks_path: str,
    interactions_path: str,
    constraints_path: str,
    out_dir: str,
    model_dir: str,
    epochs: int = 10,
    batch_size: int = 64,
    lr: float = 2e-3,
    seed: int = 42,
    lambda_constraints: float = 0.2,
    constraint_margin: float = 1.0,
    lambda_domain: float = 0.5,
    lambda_contrast: float = 0.2,
    lambda_cov: float = 0.05,
    k: int = 10,
    epsilon: float = 0.1,
    iters: int = 200,
    prefer_cuda: bool = False,
) -> dict[str, Any]:
    out_dir_p = Path(out_dir)
    model_dir_p = Path(model_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)
    model_dir_p.mkdir(parents=True, exist_ok=True)

    settings = [
        {
            "name": "full",
            "constraints_path": constraints_path,
            "lambda_constraints": float(lambda_constraints),
            "lambda_domain": float(lambda_domain),
            "eval_method": "ot",
        },
        {
            "name": "no_domain",
            "constraints_path": constraints_path,
            "lambda_constraints": float(lambda_constraints),
            "lambda_domain": 0.0,
            "eval_method": "ot",
        },
        {
            "name": "no_constraints",
            "constraints_path": None,
            "lambda_constraints": 0.0,
            "lambda_domain": float(lambda_domain),
            "eval_method": "ot",
        },
    ]

    train_outputs: dict[str, str] = {}
    rows: list[dict[str, Any]] = []

    for i, cfg in enumerate(settings):
        model_out = model_dir_p / f"{cfg['name']}.pt"
        train_model(
            tracks_path=tracks_path,
            out_path=model_out,
            constraints_path=cfg["constraints_path"],
            epochs=int(epochs),
            batch_size=int(batch_size),
            lr=float(lr),
            seed=int(seed) + i,
            prefer_cuda=bool(prefer_cuda),
            lambda_constraints=float(cfg["lambda_constraints"]),
            constraint_margin=float(constraint_margin),
            lambda_domain=float(cfg["lambda_domain"]),
            lambda_contrast=float(lambda_contrast),
            lambda_cov=float(lambda_cov),
        )
        train_outputs[cfg["name"]] = str(model_out)

        ev_out = out_dir_p / f"ablation_{cfg['name']}.json"
        ev = evaluate_recommender(
            model_path=model_out,
            tracks_path=tracks_path,
            interactions_path=interactions_path,
            out_json=ev_out,
            method=str(cfg["eval_method"]),
            k=int(k),
            epsilon=float(epsilon),
            iters=int(iters),
            prefer_cuda=bool(prefer_cuda),
        )
        rows.append(_row(name=str(cfg["name"]), ev=ev))

    # OT ablation: keep full-trained model and disable OT at inference.
    no_ot_out = out_dir_p / "ablation_no_ot.json"
    no_ot_ev = evaluate_recommender(
        model_path=train_outputs["full"],
        tracks_path=tracks_path,
        interactions_path=interactions_path,
        out_json=no_ot_out,
        method="knn",
        k=int(k),
        epsilon=float(epsilon),
        iters=int(iters),
        prefer_cuda=bool(prefer_cuda),
    )
    rows.append(_row(name="no_ot", ev=no_ot_ev))

    summary = {
        "config": {
            "tracks_path": str(tracks_path),
            "interactions_path": str(interactions_path),
            "constraints_path": str(constraints_path),
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "lr": float(lr),
            "seed": int(seed),
            "lambda_constraints": float(lambda_constraints),
            "constraint_margin": float(constraint_margin),
            "lambda_domain": float(lambda_domain),
            "lambda_contrast": float(lambda_contrast),
            "lambda_cov": float(lambda_cov),
            "k": int(k),
            "epsilon": float(epsilon),
            "iters": int(iters),
            "prefer_cuda": bool(prefer_cuda),
        },
        "rows": rows,
        "models": train_outputs,
    }

    out_json = out_dir_p / "ablation_summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    _write_markdown(out_dir_p / "ablation_table_draft.md", rows)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Run domain/OT/constraints ablation and export draft table.")
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--interactions", required=True)
    ap.add_argument("--constraints", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lambda_constraints", type=float, default=0.2)
    ap.add_argument("--constraint_margin", type=float, default=1.0)
    ap.add_argument("--lambda_domain", type=float, default=0.5)
    ap.add_argument("--lambda_contrast", type=float, default=0.2)
    ap.add_argument("--lambda_cov", type=float, default=0.05)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--epsilon", type=float, default=0.1)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--prefer_cuda", action="store_true")
    args = ap.parse_args()

    out = run_ablation(
        tracks_path=str(args.tracks),
        interactions_path=str(args.interactions),
        constraints_path=str(args.constraints),
        out_dir=str(args.out_dir),
        model_dir=str(args.model_dir),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        seed=int(args.seed),
        lambda_constraints=float(args.lambda_constraints),
        constraint_margin=float(args.constraint_margin),
        lambda_domain=float(args.lambda_domain),
        lambda_contrast=float(args.lambda_contrast),
        lambda_cov=float(args.lambda_cov),
        k=int(args.k),
        epsilon=float(args.epsilon),
        iters=int(args.iters),
        prefer_cuda=bool(args.prefer_cuda),
    )
    print(json.dumps({"rows": out["rows"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()

