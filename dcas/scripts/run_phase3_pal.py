from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dcas.pipelines import pal_tasks, train_model
from dcas.scripts.build_pal_feedback_constraints import build_constraints
from dcas.scripts.evaluate_recommender import evaluate_recommender


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _save_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _merge_constraints(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[tuple[str, str], dict[str, Any]] = {}
    for r in rows:
        a = str(r["track_id_a"]).strip()
        b = str(r["track_id_b"]).strip()
        if not a or not b or a == b:
            continue
        key = (a, b) if a < b else (b, a)
        merged[key] = {
            "track_id_a": key[0],
            "track_id_b": key[1],
            "similar": bool(r.get("similar", False)),
            "rationale": r.get("rationale", ""),
        }
    out = list(merged.values())
    out.sort(key=lambda x: (x["track_id_a"], x["track_id_b"]))
    return out


def _summary_row(tag: str, ev: dict[str, Any]) -> dict[str, Any]:
    s = ev["summary"]
    return {
        "tag": tag,
        "n_users": int(s["n_users"]),
        "n_cultures": int(s["n_cultures"]),
        "n_user_culture_evals": int(s["n_user_culture_evals"]),
        "serendipity_mean": float(s["serendipity_mean"]),
        "cultural_calibration_kl_mean": float(s["cultural_calibration_kl_mean"]),
    }


def _write_markdown(path: str | Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    base = rows[0]
    lines: list[str] = []
    lines.append("# Phase 3 PAL Two-Round Gain Report")
    lines.append("")
    lines.append("| run | serendipity_mean | delta_vs_baseline | cultural_calibration_kl_mean | delta_vs_baseline | evals |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for r in rows:
        d_ser = float(r["serendipity_mean"]) - float(base["serendipity_mean"])
        d_kl = float(r["cultural_calibration_kl_mean"]) - float(base["cultural_calibration_kl_mean"])
        lines.append(
            f"| {r['tag']} | {r['serendipity_mean']:.10f} | {d_ser:+.10f} | "
            f"{r['cultural_calibration_kl_mean']:.10f} | {d_kl:+.10f} | {int(r['n_user_culture_evals'])} |"
        )
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_phase3_pal(
    tracks_path: str,
    interactions_path: str,
    metadata_csv: str,
    baseline_model_path: str,
    out_dir: str,
    artifacts_dir: str,
    model_dir: str,
    rounds: int = 2,
    tasks_per_round: int = 120,
    label_col: str = "label",
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
    artifacts_p = Path(artifacts_dir)
    model_dir_p = Path(model_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)
    artifacts_p.mkdir(parents=True, exist_ok=True)
    model_dir_p.mkdir(parents=True, exist_ok=True)

    baseline_eval_path = out_dir_p / "eval_round0_baseline.json"
    baseline_eval = evaluate_recommender(
        model_path=baseline_model_path,
        tracks_path=tracks_path,
        interactions_path=interactions_path,
        out_json=baseline_eval_path,
        method="ot",
        k=int(k),
        epsilon=float(epsilon),
        iters=int(iters),
        prefer_cuda=bool(prefer_cuda),
    )

    rows: list[dict[str, Any]] = [_summary_row(tag="baseline", ev=baseline_eval)]
    rounds_info: list[dict[str, Any]] = []
    all_constraints: list[dict[str, Any]] = []
    current_model_path = str(baseline_model_path)

    for ridx in range(1, int(rounds) + 1):
        tasks_path = artifacts_p / f"round{ridx}_pal_tasks.jsonl"
        task_info = pal_tasks(
            model_path=current_model_path,
            tracks_path=tracks_path,
            out_path=tasks_path,
            n=int(tasks_per_round),
            prefer_cuda=bool(prefer_cuda),
        )

        round_constraints_path = artifacts_p / f"round{ridx}_constraints.jsonl"
        round_constraints_report = build_constraints(
            tasks_path=str(tasks_path),
            metadata_csv=metadata_csv,
            out_path=str(round_constraints_path),
            track_id_col="track_id",
            label_col=str(label_col),
        )
        round_constraints = _load_jsonl(round_constraints_path)
        all_constraints.extend(round_constraints)
        merged = _merge_constraints(all_constraints)

        merged_path = artifacts_p / f"constraints_upto_round{ridx}.jsonl"
        _save_jsonl(merged_path, merged)

        model_out = model_dir_p / f"round{ridx}_model.pt"
        train_info = train_model(
            tracks_path=tracks_path,
            out_path=model_out,
            constraints_path=merged_path,
            epochs=int(epochs),
            batch_size=int(batch_size),
            lr=float(lr),
            seed=int(seed) + ridx,
            prefer_cuda=bool(prefer_cuda),
            lambda_constraints=float(lambda_constraints),
            constraint_margin=float(constraint_margin),
            lambda_domain=float(lambda_domain),
            lambda_contrast=float(lambda_contrast),
            lambda_cov=float(lambda_cov),
        )

        eval_path = out_dir_p / f"eval_round{ridx}.json"
        eval_info = evaluate_recommender(
            model_path=model_out,
            tracks_path=tracks_path,
            interactions_path=interactions_path,
            out_json=eval_path,
            method="ot",
            k=int(k),
            epsilon=float(epsilon),
            iters=int(iters),
            prefer_cuda=bool(prefer_cuda),
        )

        rows.append(_summary_row(tag=f"round{ridx}", ev=eval_info))
        rounds_info.append(
            {
                "round": int(ridx),
                "task_info": task_info,
                "round_constraints_report": round_constraints_report,
                "n_merged_constraints": int(len(merged)),
                "train_history_tail": train_info["history"][-3:],
                "model_path": str(model_out),
                "eval_path": str(eval_path),
            }
        )
        current_model_path = str(model_out)

    summary = {
        "config": {
            "tracks_path": str(tracks_path),
            "interactions_path": str(interactions_path),
            "metadata_csv": str(metadata_csv),
            "baseline_model_path": str(baseline_model_path),
            "rounds": int(rounds),
            "tasks_per_round": int(tasks_per_round),
            "label_col": str(label_col),
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
        "rounds": rounds_info,
    }

    out_json = out_dir_p / "phase3_pal_summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    _write_markdown(out_dir_p / "phase3_pal_summary.md", rows)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Run phase-3 PAL two-round feedback and gain comparison.")
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--interactions", required=True)
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--baseline_model", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--artifacts_dir", required=True)
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--tasks_per_round", type=int, default=120)
    ap.add_argument("--label_col", default="label")
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

    out = run_phase3_pal(
        tracks_path=str(args.tracks),
        interactions_path=str(args.interactions),
        metadata_csv=str(args.metadata),
        baseline_model_path=str(args.baseline_model),
        out_dir=str(args.out_dir),
        artifacts_dir=str(args.artifacts_dir),
        model_dir=str(args.model_dir),
        rounds=int(args.rounds),
        tasks_per_round=int(args.tasks_per_round),
        label_col=str(args.label_col),
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

