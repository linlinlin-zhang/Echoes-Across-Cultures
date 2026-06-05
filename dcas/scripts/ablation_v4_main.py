#!/usr/bin/env python
"""
V4 Main Ablation Study — Rebuild on current V4 data with GPU.

Trains three model variants (full, no_domain, no_constraints) + no_OT inference
on V4 main dataset, using the existing train_model pipeline.

Outputs:
  - Per-culture breakdown tables
  - Paired bootstrap CIs (1000 samples)
  - Manuscript-ready Markdown tables
  - Per-culture visualization PNG

Usage (GPU):
    python -m dcas.scripts.ablation_v4_main --gpu
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from dcas.data.interactions import load_interactions
from dcas.data.npz_tracks import load_tracks
from dcas.pipelines import train_model
from dcas.recommender import recommend_ot, recommend_ot_calibrated, recommend_knn
from dcas.serialization import load_checkpoint

# ── V4 Main paths ──────────────────────────────────────────────────────
V4_CM_TRACKS = "storage/public/research_dataset_v4/main/tracks_culturemert_mw3.npz"
V4_CM_INTER = "storage/public/research_dataset_v4/main/interactions_synth_mixed_culturemert_mw3.csv"
V4_GEM_TRACKS = "storage/public/research_dataset_v4/main/tracks_gemini_embedding2_mw3.npz"
V4_GEM_INTER = "storage/public/research_dataset_v4/main/interactions_synth_mixed_gemini_embedding2_mw3.csv"
CONSTRAINTS = "storage/pal/v2_main_gemini_simulated/constraints_upto_round1.jsonl"

OUT_DIR = Path("reports/audits/ablation_v4_main_2026-04-05")
MODEL_DIR = OUT_DIR / "models"

# Training config matching V4 stage-3
EPOCHS = 10
BATCH_SIZE = 128
LR = 2e-3
SEED = 42

# Eval config
K = 10
EPSILON = 0.1
ITERS = 200
BOOTSTRAP = 1000


def _compute_minority(tracks, interactions, recs, k=K, quantile=0.25):
    """Compute minority_exposure_at_k for a list of recommendations."""
    if not recs:
        return float("nan")
    pop = {}
    for it in interactions:
        tid = str(it.track_id)
        pop[tid] = pop.get(tid, 0.0) + float(it.weight)
    all_ids = [str(tid) for tid in tracks.track_id.tolist()]
    pop_arr = np.array([pop.get(tid, 0.0) for tid in all_ids], dtype=np.float64)
    if float(np.max(pop_arr) - np.min(pop_arr)) <= 1e-12:
        n_minority = max(1, min(len(all_ids), int(round(len(all_ids) * quantile))))
        order = np.argsort(np.array(all_ids, dtype=object))
        minority = {all_ids[i] for i in order[:n_minority]}
    else:
        threshold = float(np.quantile(pop_arr, quantile))
        minority = {tid for tid, p in zip(all_ids, pop_arr.tolist()) if p <= threshold}
        if not minority:
            minority = {all_ids[int(np.argmin(pop_arr))]}
    hit = sum(1 for r in recs if str(r.track_id) in minority)
    return float(hit / len(recs))


def _eval_all_users(model_path, tracks_path, inter_path, device, method="ot_calibrated"):
    """Evaluate over all users x cultures, return rows list."""
    model, _ = load_checkpoint(model_path, map_location=str(device))
    tracks = load_tracks(tracks_path)
    interactions = load_interactions(inter_path)
    users = sorted({str(i.user_id) for i in interactions})
    cultures = tracks.cultures()
    rows = []

    CAL_W = dict(
        relevance_weight=0.48,
        novelty_weight=0.10,
        target_affinity_weight=0.22,
        minority_weight=0.14,
        source_weight=0.06,
        diversity_lambda=0.03,
    )

    for u in users:
        for c in cultures:
            try:
                if method == "knn":
                    recs, metrics = recommend_knn(
                        model=model,
                        tracks=tracks,
                        interactions=interactions,
                        user_id=u,
                        target_culture=c,
                        k=K,
                        device=device,
                    )
                elif method == "ot_calibrated":
                    recs, metrics = recommend_ot_calibrated(
                        model=model,
                        tracks=tracks,
                        interactions=interactions,
                        user_id=u,
                        target_culture=c,
                        k=K,
                        device=device,
                        epsilon=EPSILON,
                        iters=ITERS,
                        **CAL_W,
                    )
                else:
                    recs, metrics = recommend_ot(
                        model=model,
                        tracks=tracks,
                        interactions=interactions,
                        user_id=u,
                        target_culture=c,
                        k=K,
                        device=device,
                        epsilon=EPSILON,
                        iters=ITERS,
                    )
                min_exp = _compute_minority(tracks, interactions, recs, k=K)
                rows.append(
                    {
                        "user_id": u,
                        "target_culture": c,
                        "serendipity": float(metrics["serendipity"]),
                        "cultural_calibration_kl": float(metrics["cultural_calibration_kl"]),
                        "minority_exposure_at_k": min_exp,
                    }
                )
            except Exception:
                pass
    return rows


def _summary(rows):
    ser = [r["serendipity"] for r in rows]
    ckl = [r["cultural_calibration_kl"] for r in rows]
    mn = [r["minority_exposure_at_k"] for r in rows if not np.isnan(r["minority_exposure_at_k"])]
    return {
        "n": len(rows),
        "serendipity_mean": float(np.mean(ser)),
        "cultural_calibration_kl_mean": float(np.mean(ckl)),
        "minority_exposure_at_k_mean": float(np.mean(mn)) if mn else float("nan"),
    }


def _per_culture(rows, cultures):
    out = {}
    for c in cultures:
        cr = [r for r in rows if r["target_culture"] == c]
        if not cr:
            continue
        mn = [r["minority_exposure_at_k"] for r in cr if not np.isnan(r["minority_exposure_at_k"])]
        out[c] = {
            "n": len(cr),
            "serendipity_mean": float(np.mean([r["serendipity"] for r in cr])),
            "cultural_calibration_kl_mean": float(np.mean([r["cultural_calibration_kl"] for r in cr])),
            "minority_exposure_at_k_mean": float(np.mean(mn)) if mn else float("nan"),
        }
    return out


def _bootstrap_ci(deltas, samples=1000, seed=42):
    if len(deltas) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(deltas), size=(samples, len(deltas)))
    means = np.array(deltas)[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def _perm_pval(deltas, samples=1000, seed=42):
    if len(deltas) == 0:
        return float("nan")
    obs = np.mean(deltas)
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(samples, len(deltas)))
    perm = (signs * np.array(deltas)[None, :]).mean(axis=1)
    return float((np.sum(np.abs(perm) >= abs(obs)) + 1) / (samples + 1))


def run_ablation(backbone="culturemert", device=None, skip_train=False):
    import torch

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tracks_path = V4_CM_TRACKS if backbone == "culturemert" else V4_GEM_TRACKS
    inter_path = V4_CM_INTER if backbone == "culturemert" else V4_GEM_INTER
    prefix = f"v4_main_{backbone}"
    out_dir = OUT_DIR / prefix
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir = out_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    tracks = load_tracks(tracks_path)
    cultures = tracks.cultures()

    # ── Phase 1: Train variants ────────────────────────────────────────
    if not skip_train:
        variants = [
            {"name": "full", "lambda_domain": 0.50, "constraints": CONSTRAINTS},
            {"name": "no_domain", "lambda_domain": 0.0, "constraints": CONSTRAINTS},
            {"name": "no_constraints", "lambda_domain": 0.50, "constraints": None},
        ]
        train_outputs = {}
        for v in variants:
            t0 = time.time()
            mp = str(model_dir / f"{v['name']}.pt")
            print(f"\n[{prefix}] Training {v['name']}...")
            h = train_model(
                tracks_path=tracks_path,
                out_path=mp,
                constraints_path=v["constraints"],
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                lr=LR,
                seed=SEED,
                prefer_cuda=device.type == "cuda",
                lambda_domain=v["lambda_domain"],
                lambda_contrast=0.20,
                lambda_cov=0.05,
                lambda_tc=0.05,
                lambda_hsic=0.02,
                lambda_source=0.10,
                beta_kl=1.0,
                lambda_constraints=0.15,
                constraint_margin=1.0,
                source_balanced_batch=True,
                regularizer_warmup_epochs=3,
                constraint_start_epoch=2,
                constraint_warmup_epochs=2,
                rank_start_epoch=4,
                rank_warmup_epochs=2,
                lambda_rank=0.12,
                interactions_path=inter_path,
            )
            train_outputs[v["name"]] = mp
            print(f"  [{v['name']}] Done in {time.time() - t0:.0f}s, final loss={h['history'][-1]['loss']:.6f}")
    else:
        # Use existing models
        train_outputs = {}
        for name in ["full", "no_domain", "no_constraints"]:
            mp = str(model_dir / f"{name}.pt")
            if Path(mp).exists():
                train_outputs[name] = mp
                print(f"  [{name}] Found existing model: {mp}")
            else:
                print(f"  [{name}] NOT FOUND: {mp}")

    if not train_outputs:
        raise RuntimeError("No models to evaluate")

    # ── Phase 2: Evaluate all variants ─────────────────────────────────
    print(f"\n[{prefix}] Evaluating (ot_calibrated mode)...")
    eval_rows = {}
    for name, mp in train_outputs.items():
        rows = _eval_all_users(mp, tracks_path, inter_path, device, method="ot_calibrated")
        eval_rows[name] = rows
        s = _summary(rows)
        print(
            f"  {name}: ser={s['serendipity_mean']:.4f}, ckl={s['cultural_calibration_kl_mean']:.4f}, min={s['minority_exposure_at_k_mean']:.4f}"
        )

    # no_OT (knn calibrated) on full model
    if "full" in train_outputs:
        rows_no_ot = _eval_all_users(train_outputs["full"], tracks_path, inter_path, device, method="knn")
        eval_rows["no_ot"] = rows_no_ot
        s = _summary(rows_no_ot)
        print(
            f"  no_ot(knn): ser={s['serendipity_mean']:.4f}, ckl={s['cultural_calibration_kl_mean']:.4f}, min={s['minority_exposure_at_k_mean']:.4f}"
        )

    # Save raw eval JSONs
    for name, rows in eval_rows.items():
        ev = {
            "summary": _summary(rows),
            "per_target_culture": _per_culture(rows, cultures),
            "rows": rows,
        }
        ep = out_dir / f"eval_{name}.json"
        with open(ep, "w", encoding="utf-8") as f:
            json.dump(ev, f, indent=2)

    # ── Phase 3: Paired comparisons ────────────────────────────────────
    print(f"\n[{prefix}] Paired comparisons vs full...")
    full_rows = eval_rows["full"]
    comparisons = {}
    for vname in ["no_domain", "no_constraints", "no_ot"]:
        if vname not in eval_rows:
            continue
        cand_rows = eval_rows[vname]
        comp = {"metrics": {}, "per_target_culture": {}}
        # Build paired deltas
        for metric in [
            "serendipity",
            "cultural_calibration_kl",
            "minority_exposure_at_k",
        ]:
            deltas = []
            keys = []
            for fr, cr in zip(full_rows, cand_rows):
                df = fr.get(metric, float("nan"))
                dc = cr.get(metric, float("nan"))
                if np.isnan(df) or np.isnan(dc):
                    continue
                deltas.append(dc - df)
                keys.append((fr["user_id"], fr["target_culture"]))
            da = np.array(deltas, dtype=np.float64)
            ci_lo, ci_hi = _bootstrap_ci(deltas, BOOTSTRAP, 42)
            pv = _perm_pval(deltas, BOOTSTRAP, 142)
            base_vals = [
                fr[metric]
                for fr, cr in zip(full_rows, cand_rows)
                if not np.isnan(fr.get(metric, float("nan"))) and not np.isnan(cr.get(metric, float("nan")))
            ]
            cand_vals = [
                cr[metric]
                for fr, cr in zip(full_rows, cand_rows)
                if not np.isnan(fr.get(metric, float("nan"))) and not np.isnan(cr.get(metric, float("nan")))
            ]
            comp["metrics"][metric] = {
                "n_pairs": int(len(deltas)),
                "base_mean": float(np.mean(base_vals)) if base_vals else float("nan"),
                "candidate_mean": float(np.mean(cand_vals)) if cand_vals else float("nan"),
                "delta_mean": float(da.mean()),
                "delta_ci95_low": ci_lo,
                "delta_ci95_high": ci_hi,
                "p_value_two_sided": pv,
            }
            # Per culture
            for c in cultures:
                cd = [d for (uid, tc), d in zip(keys, deltas) if tc == c]
                if cd:
                    cda = np.array(cd, dtype=np.float64)
                    cci = _bootstrap_ci(cd, BOOTSTRAP, 242)
                    cpv = _perm_pval(cd, BOOTSTRAP, 342)
                    comp["per_target_culture"].setdefault(c, {})[metric] = {
                        "delta_mean": float(cda.mean()),
                        "delta_ci95_low": cci[0],
                        "delta_ci95_high": cci[1],
                        "p_value_two_sided": cpv,
                    }
        comparisons[vname] = comp
        cp = out_dir / f"comparison_full_vs_{vname}.json"
        with open(cp, "w", encoding="utf-8") as f:
            json.dump(comp, f, indent=2)

    # ── Phase 4: Manuscript tables ─────────────────────────────────────
    full_s = _summary(full_rows)
    lines = [
        f"# V4 Main Ablation — {backbone}",
        "",
        f"k={K} | bootstrap={BOOTSTRAP} | seed={SEED}",
        "",
        "| Variant | Serendipity ↑ | Δ vs Full | 95% CI | p-value | Calib KL ↓ | Minority@K ↑ |",
        "|---------|--------------:|----------:|-------:|--------:|-----------:|-------------:|",
    ]
    for vname in ["full", "no_domain", "no_constraints", "no_ot"]:
        if vname not in eval_rows:
            continue
        s = _summary(eval_rows[vname])
        if vname == "full":
            lines.append(
                f"| **{vname}** | {s['serendipity_mean']:.4f} | — | — | — | {s['cultural_calibration_kl_mean']:.4f} | {s['minority_exposure_at_k_mean']:.4f} |"
            )
        else:
            m = comparisons.get(vname, {}).get("metrics", {}).get("serendipity", {})
            ci = f"[{m.get('delta_ci95_low', 0):.4f}, {m.get('delta_ci95_high', 0):.4f}]"
            pv = f"{m.get('p_value_two_sided', 0):.6f}"
            delta = s["serendipity_mean"] - full_s["serendipity_mean"]
            lines.append(
                f"| {vname} | {s['serendipity_mean']:.4f} | {delta:+.4f} | {ci} | {pv} | {s['cultural_calibration_kl_mean']:.4f} | {s['minority_exposure_at_k_mean']:.4f} |"
            )

    with open(out_dir / "ablation_table.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    # Per-culture table
    cl = [
        "# Per-Culture Ablation Breakdown",
        "",
        "| Culture | Variant | Δ Ser ↑ | 95% CI | p-value | Δ CKL ↓ | p-value | Δ Min@K ↑ | p-value |",
        "|---------|---------|--------:|-------:|--------:|--------:|--------:|----------:|--------:|",
    ]
    for c in cultures:
        for vn in ["no_domain", "no_constraints", "no_ot"]:
            ptc = comparisons.get(vn, {}).get("per_target_culture", {}).get(c, {})
            vals = []
            for mt in [
                "serendipity",
                "cultural_calibration_kl",
                "minority_exposure_at_k",
            ]:
                m = ptc.get(mt, {})
                d = m.get("delta_mean", 0)
                ci_l = m.get("delta_ci95_low", 0)
                ci_h = m.get("delta_ci95_high", 0)
                pv = m.get("p_value_two_sided", 0)
                vals.append((d, ci_l, ci_h, pv))
            parts = []
            for d, ci_l, ci_h, pv in vals:
                parts.append(f"{d:+.4f}")
                parts.append(f"[{ci_l:.4f}, {ci_h:.4f}]")
                parts.append(f"{pv:.4f}")
            cl.append(f"| {c} | {vn} | " + " | ".join(parts) + " |")

    with open(out_dir / "ablation_per_culture.md", "w", encoding="utf-8") as f:
        f.write("\n".join(cl) + "\n")

    # ── Phase 5: Per-culture visualization ─────────────────────────────
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        metrics_map = {
            "Serendipity (Δ)": "serendipity",
            "Calibration KL (Δ)": "cultural_calibration_kl",
            "Minority Exposure (Δ)": "minority_exposure_at_k",
        }
        colors = {
            "no_domain": "#e74c3c",
            "no_constraints": "#f39c12",
            "no_ot": "#3498db",
        }
        for ax, (title, metric) in zip(axes, metrics_map.items()):
            x = np.arange(len(cultures))
            width = 0.25
            for i, vn in enumerate(["no_domain", "no_constraints", "no_ot"]):
                ptc = comparisons.get(vn, {}).get("per_target_culture", {})
                vals = [ptc.get(c, {}).get(metric, {}).get("delta_mean", 0) for c in cultures]
                cis_lo = [ptc.get(c, {}).get(metric, {}).get("delta_ci95_low", 0) for c in cultures]
                cis_hi = [ptc.get(c, {}).get(metric, {}).get("delta_ci95_high", 0) for c in cultures]
                ax.bar(x + i * width, vals, width, label=vn, color=colors[vn], alpha=0.85)
                ax.errorbar(
                    x + i * width,
                    vals,
                    yerr=[
                        np.abs(np.array(vals) - np.array(cis_lo)),
                        np.abs(np.array(cis_hi) - np.array(vals)),
                    ],
                    fmt="none",
                    color=colors[vn],
                    capsize=3,
                    alpha=0.7,
                )
            ax.set_xticks(x + width)
            ax.set_xticklabels(cultures, rotation=30, ha="right", fontsize=8)
            ax.set_ylabel(title, fontsize=9)
            ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
            ax.legend(fontsize=7)
            ax.grid(axis="y", alpha=0.3)

        plt.suptitle(f"V4 Main Ablation — {backbone} (Δ vs Full)", fontsize=11, fontweight="bold")
        plt.tight_layout()
        png_path = out_dir / "ablation_per_culture.png"
        plt.savefig(png_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"\n  Figure saved: {png_path}")
    except Exception as e:
        print(f"\n  Figure generation failed: {e}")

    # Summary JSON
    summary = {
        "backbone": backbone,
        "device": str(device),
        "bootstrap_samples": BOOTSTRAP,
        "eval_summaries": {k: _summary(v) for k, v in eval_rows.items()},
        "n_comparisons": len(comparisons),
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"V4 Main Ablation ({backbone}) complete")
    print(f"  Table: {out_dir / 'ablation_table.md'}")
    print(f"  Per-culture: {out_dir / 'ablation_per_culture.md'}")
    print(f"  Figure: {out_dir / 'ablation_per_culture.png'}")
    print(f"{'=' * 60}")
    return summary


if __name__ == "__main__":
    import argparse
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="culturemert", choices=["culturemert", "gemini"])
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument(
        "--skip_train",
        action="store_true",
        help="Use existing models, only eval+compare",
    )
    args = ap.parse_args()
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    print(f"Device: {device}, backbone: {args.backbone}, skip_train: {args.skip_train}")
    run_ablation(backbone=args.backbone, device=device, skip_train=args.skip_train)
