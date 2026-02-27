# Unified Eval Suite Guide

Date: 2026-02-27

This guide describes the unified non-dataset evaluation entry:

- `python -m dcas.cli.eval`

It runs:
1. Recommender evaluation (with bootstrap CI)
2. Optional paired significance test against a baseline model
3. Optional disentanglement evaluation (MIG/DCI/SAP)

## 1) Recommender-only

```bash
python -m dcas.cli.eval \
  --model ./storage/public/routeA_phase4_disentangle_cn/model_tc_hsic.pt \
  --tracks ./storage/public/routeA_phase2_cn/tracks.npz \
  --interactions ./storage/public/routeA_phase2_cn/interactions.csv \
  --out_dir ./reports/eval_suite_phase4_only \
  --method ot \
  --k 10 \
  --epsilon 0.1 \
  --iters 200 \
  --bootstrap_samples 5000 \
  --permutation_samples 5000
```

## 2) Recommender + baseline significance

```bash
python -m dcas.cli.eval \
  --model ./storage/public/routeA_phase4_disentangle_cn/model_tc_hsic.pt \
  --baseline_model ./storage/public/routeA_phase2_cn/model.pt \
  --tracks ./storage/public/routeA_phase2_cn/tracks.npz \
  --interactions ./storage/public/routeA_phase2_cn/interactions.csv \
  --out_dir ./reports/eval_suite_phase4_vs_phase2 \
  --method ot \
  --k 10 \
  --epsilon 0.1 \
  --iters 200 \
  --bootstrap_samples 5000 \
  --permutation_samples 5000
```

## 3) Full suite (recommender + significance + disentanglement)

```bash
python -m dcas.cli.eval \
  --model ./storage/public/routeA_phase4_disentangle_cn/model_tc_hsic.pt \
  --baseline_model ./storage/public/routeA_phase2_cn/model.pt \
  --tracks ./storage/public/routeA_phase2_cn/tracks.npz \
  --interactions ./storage/public/routeA_phase2_cn/interactions.csv \
  --metadata ./reports/routeA_shared_factors_cn.csv \
  --factors factor_energy,factor_brightness,factor_texture,factor_dynamics,factor_percussiveness \
  --seeds 42,43,44,45,46 \
  --out_dir ./reports/eval_suite_phase4_full \
  --method ot \
  --k 10 \
  --epsilon 0.1 \
  --iters 200 \
  --bootstrap_samples 5000 \
  --permutation_samples 5000
```

## Output artifacts

- `recommender_eval.json`
- `recommender_eval_baseline.json` (if baseline model provided)
- `recommender_compare.{json,md}` (if baseline model provided)
- `disentanglement_eval.{json,md}` (if metadata/factors provided)
- `eval_suite_summary.json` (always)

## Notes

- `cultural_calibration_kl` now uses a smoothed target-culture prior in style latent (`zs`) space to avoid degenerate constant values.
- Legacy hard-distribution calibration is retained as `cultural_calibration_kl_legacy` for backward comparison.
