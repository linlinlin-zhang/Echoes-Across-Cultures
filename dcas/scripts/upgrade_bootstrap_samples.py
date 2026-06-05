#!/usr/bin/env python
"""
Upgrade bootstrap/permutation samples from 200 to 1000 for all V4 comparisons.

Reads existing eval JSONs and re-runs compare_recommender_runs with 1000 samples.
Outputs updated comparison JSONs.
"""

from __future__ import annotations

import json
from pathlib import Path

from dcas.scripts.compare_recommender_runs import compare_recommender_runs

BOOTSTRAP = 1000
PERMUTATION = 1000
SEED = 42


def upgrade_dir(benchmark_dir: Path) -> None:
    """Upgrade all comparison JSONs in a benchmark directory."""
    comp_dir = benchmark_dir / "comparisons"
    if not comp_dir.exists():
        return

    count = 0
    for comp_file in sorted(comp_dir.glob("*.json")):
        with open(comp_file, "r", encoding="utf-8") as f:
            old = json.load(f)

        base_path = old.get("base_eval_path", "")
        cand_path = old.get("candidate_eval_path", "")
        metrics = old.get("config", {}).get(
            "metrics",
            ["serendipity", "cultural_calibration_kl", "minority_exposure_at_k"],
        )

        if not base_path or not cand_path:
            print(f"  SKIP {comp_file.name}: missing paths")
            continue
        if not Path(base_path).exists() or not Path(cand_path).exists():
            print(f"  SKIP {comp_file.name}: files not found")
            continue

        cmp = compare_recommender_runs(
            base_eval_path=base_path,
            candidate_eval_path=cand_path,
            metrics=metrics,
            bootstrap_samples=BOOTSTRAP,
            permutation_samples=PERMUTATION,
            seed=SEED,
            out_json=comp_file,
            out_md=comp_file.with_suffix(".md"),
        )
        count += 1
        # Print summary
        for m, r in cmp.get("metrics", {}).items():
            dm = r.get("delta_mean", 0)
            ci = f"[{r.get('delta_ci95_low', 0):.4f}, {r.get('delta_ci95_high', 0):.4f}]"
            pv = f"{r.get('p_value_two_sided', 0):.6f}"
            print(f"  {comp_file.name:60s} {m:25s} Δ={dm:+.4f} {ci} p={pv}")

    print(f"  Upgraded {count} comparisons in {benchmark_dir.name}")


def main() -> None:
    base = Path("reports/benchmarks")
    dirs = sorted(base.glob("v4_*_lambdamart"))
    print(f"Found {len(dirs)} V4 benchmark directories to upgrade")
    for d in dirs:
        print(f"\nProcessing {d.name}...")
        upgrade_dir(d)
    print("\nDone.")


if __name__ == "__main__":
    main()
