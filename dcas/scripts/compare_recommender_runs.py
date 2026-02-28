from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _load_rows(path: str | Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    rows = obj.get("rows", [])
    if not isinstance(rows, list):
        raise ValueError(f"invalid rows in {path}")
    return rows


def _to_map(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for r in rows:
        user_id = str(r.get("user_id", "")).strip()
        target = str(r.get("target_culture", "")).strip()
        if not user_id or not target:
            continue
        out[(user_id, target)] = r
    return out


def _paired_arrays(
    base_map: dict[tuple[str, str], dict[str, Any]],
    cand_map: dict[tuple[str, str], dict[str, Any]],
    metric: str,
) -> tuple[list[tuple[str, str]], np.ndarray, np.ndarray]:
    keys_all = sorted(set(base_map.keys()) & set(cand_map.keys()))
    if not keys_all:
        raise ValueError(f"no overlapping (user_id,target_culture) for metric={metric}")
    keys: list[tuple[str, str]] = []
    bvals: list[float] = []
    cvals: list[float] = []
    for k in keys_all:
        br = base_map[k]
        cr = cand_map[k]
        if metric not in br or metric not in cr:
            continue
        try:
            b = float(br[metric])
            c = float(cr[metric])
        except Exception:
            continue
        if np.isnan(b) or np.isnan(c):
            continue
        keys.append(k)
        bvals.append(b)
        cvals.append(c)
    if not keys:
        raise ValueError(f"no valid overlapping rows contain metric={metric}")
    base = np.array(bvals, dtype=np.float64)
    cand = np.array(cvals, dtype=np.float64)
    return keys, base, cand


def _bootstrap_ci(delta: np.ndarray, samples: int, seed: int) -> tuple[float, float]:
    if delta.size == 0:
        return float("nan"), float("nan")
    if delta.size < 2 or int(samples) <= 0:
        m = float(delta.mean())
        return m, m
    rng = np.random.default_rng(int(seed))
    idx = rng.integers(0, int(delta.size), size=(int(samples), int(delta.size)))
    means = delta[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def _permutation_pvalue(delta: np.ndarray, samples: int, seed: int) -> float:
    if delta.size == 0:
        return float("nan")
    if int(samples) <= 0:
        return float("nan")
    obs = float(delta.mean())
    rng = np.random.default_rng(int(seed))
    signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float64), size=(int(samples), int(delta.size)))
    perm_means = (signs * delta[None, :]).mean(axis=1)
    p = (float(np.sum(np.abs(perm_means) >= abs(obs))) + 1.0) / float(int(samples) + 1)
    return float(p)


def _cohen_d_paired(delta: np.ndarray) -> float:
    if delta.size == 0:
        return float("nan")
    if delta.size == 1:
        return 0.0
    sd = float(np.std(delta, ddof=1))
    if sd <= 1e-12:
        return 0.0
    return float(np.mean(delta) / sd)


def _culture_breakdown(
    keys: list[tuple[str, str]],
    base: np.ndarray,
    cand: np.ndarray,
    bootstrap_samples: int,
    permutation_samples: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    idx_by_culture: dict[str, list[int]] = {}
    for i, (_, c) in enumerate(keys):
        idx_by_culture.setdefault(str(c), []).append(int(i))

    out: dict[str, dict[str, float]] = {}
    for c in sorted(idx_by_culture.keys()):
        idx = np.array(idx_by_culture[c], dtype=np.int64)
        b = base[idx]
        ca = cand[idx]
        d = ca - b
        lo, hi = _bootstrap_ci(d, samples=int(bootstrap_samples), seed=int(seed) + 7)
        p = _permutation_pvalue(d, samples=int(permutation_samples), seed=int(seed) + 17)
        out[c] = {
            "n_pairs": int(idx.size),
            "base_mean": float(b.mean()) if b.size else float("nan"),
            "candidate_mean": float(ca.mean()) if ca.size else float("nan"),
            "delta_mean": float(d.mean()) if d.size else float("nan"),
            "delta_ci95_low": float(lo),
            "delta_ci95_high": float(hi),
            "p_value_two_sided": float(p),
            "cohen_d_paired": float(_cohen_d_paired(d)),
        }
    return out


def compare_recommender_runs(
    base_eval_path: str | Path,
    candidate_eval_path: str | Path,
    metrics: list[str] | None = None,
    bootstrap_samples: int = 5000,
    permutation_samples: int = 5000,
    seed: int = 42,
    out_json: str | Path | None = None,
    out_md: str | Path | None = None,
) -> dict[str, Any]:
    metric_list = metrics or ["serendipity", "cultural_calibration_kl"]
    metric_list = [str(m).strip() for m in metric_list if str(m).strip()]
    if not metric_list:
        raise ValueError("metrics cannot be empty")

    base_rows = _load_rows(base_eval_path)
    cand_rows = _load_rows(candidate_eval_path)
    base_map = _to_map(base_rows)
    cand_map = _to_map(cand_rows)

    result: dict[str, Any] = {
        "base_eval_path": str(base_eval_path),
        "candidate_eval_path": str(candidate_eval_path),
        "config": {
            "metrics": metric_list,
            "bootstrap_samples": int(bootstrap_samples),
            "permutation_samples": int(permutation_samples),
            "seed": int(seed),
        },
        "metrics": {},
    }

    n_pairs_ref: int | None = None
    skipped_metrics: dict[str, str] = {}
    used_metrics: list[str] = []
    for i, metric in enumerate(metric_list):
        try:
            keys, base, cand = _paired_arrays(base_map=base_map, cand_map=cand_map, metric=metric)
        except ValueError as e:
            skipped_metrics[metric] = str(e)
            continue
        delta = cand - base
        ci_lo, ci_hi = _bootstrap_ci(delta, samples=int(bootstrap_samples), seed=int(seed) + i * 101)
        p_value = _permutation_pvalue(delta, samples=int(permutation_samples), seed=int(seed) + i * 101 + 1)
        by_culture = _culture_breakdown(
            keys=keys,
            base=base,
            cand=cand,
            bootstrap_samples=int(bootstrap_samples),
            permutation_samples=int(permutation_samples),
            seed=int(seed) + i * 101 + 2,
        )
        result["metrics"][metric] = {
            "n_pairs": int(len(keys)),
            "base_mean": float(base.mean()),
            "candidate_mean": float(cand.mean()),
            "delta_mean": float(delta.mean()),
            "delta_std": float(np.std(delta, ddof=1)) if delta.size > 1 else 0.0,
            "delta_ci95_low": float(ci_lo),
            "delta_ci95_high": float(ci_hi),
            "p_value_two_sided": float(p_value),
            "cohen_d_paired": float(_cohen_d_paired(delta)),
            "per_target_culture": by_culture,
        }
        used_metrics.append(metric)
        if n_pairs_ref is None:
            n_pairs_ref = int(len(keys))

    result["n_pairs"] = int(n_pairs_ref or 0)
    result["config"]["used_metrics"] = used_metrics
    if skipped_metrics:
        result["config"]["skipped_metrics"] = skipped_metrics

    if out_json is not None:
        p = Path(out_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    if out_md is not None:
        lines = [
            "# Recommender Run Comparison (Paired Bootstrap + Permutation Test)",
            "",
            f"- base: `{base_eval_path}`",
            f"- candidate: `{candidate_eval_path}`",
            f"- paired samples: `{int(result['n_pairs'])}`",
            "",
            "| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for metric in used_metrics:
            r = result["metrics"].get(metric, {})
            if not r:
                continue
            ci = f"[{r['delta_ci95_low']:.6f}, {r['delta_ci95_high']:.6f}]"
            lines.append(
                f"| {metric} | {r['base_mean']:.10f} | {r['candidate_mean']:.10f} | "
                f"{r['delta_mean']:+.10f} | {ci} | {r['p_value_two_sided']:.6f} | {r['cohen_d_paired']:.6f} |"
            )
        if skipped_metrics:
            lines.append("")
            lines.append("Skipped metrics:")
            for m in metric_list:
                if m in skipped_metrics:
                    lines.append(f"- `{m}`: {skipped_metrics[m]}")
        lines.append("")
        p = Path(out_md)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare two recommender eval json files with paired significance tests.")
    ap.add_argument("--base_json", required=True)
    ap.add_argument("--candidate_json", required=True)
    ap.add_argument("--metrics", default="serendipity,cultural_calibration_kl")
    ap.add_argument("--bootstrap_samples", type=int, default=5000)
    ap.add_argument("--permutation_samples", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_json", default=None)
    ap.add_argument("--out_md", default=None)
    args = ap.parse_args()

    metrics = [x.strip() for x in str(args.metrics).split(",") if x.strip()]
    out = compare_recommender_runs(
        base_eval_path=str(args.base_json),
        candidate_eval_path=str(args.candidate_json),
        metrics=metrics,
        bootstrap_samples=int(args.bootstrap_samples),
        permutation_samples=int(args.permutation_samples),
        seed=int(args.seed),
        out_json=str(args.out_json) if args.out_json else None,
        out_md=str(args.out_md) if args.out_md else None,
    )

    tiny = {
        "n_pairs": out["n_pairs"],
        "metrics": {
            m: {
                "delta_mean": out["metrics"][m]["delta_mean"],
                "delta_ci95_low": out["metrics"][m]["delta_ci95_low"],
                "delta_ci95_high": out["metrics"][m]["delta_ci95_high"],
                "p_value_two_sided": out["metrics"][m]["p_value_two_sided"],
            }
            for m in metrics
            if m in out["metrics"]
        },
    }
    if "skipped_metrics" in out["config"]:
        tiny["skipped_metrics"] = out["config"]["skipped_metrics"]
    print(json.dumps(tiny, ensure_ascii=False))


if __name__ == "__main__":
    main()
