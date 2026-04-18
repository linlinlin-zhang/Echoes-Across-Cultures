# V4 Main CultureMERT Real PAL Stage3 Calibration Sweep

## Purpose

This sweep evaluates calibrated rerank operating points on top of the migrated real PAL checkpoint:

- checkpoint: `storage/models/pal/v4_main_culturemert_real_from_v4_main_annotation_stage3/real_pal_model.pt`
- config: `configs/benchmark/recommender_benchmark_v4_main_culturemert_real_pal_stage3_calibration_sweep.run.json`
- summary: `benchmark_summary.json`

The goal is not to retrain the model again, but to test whether the current PAL checkpoint can outperform the existing `V4 main + CultureMERT stage3` reference under a better calibration setting.

## Best Operating Points

### `pal_ot_cal_p3_balanced`

Compared with the current reference `dcas_full_ot_calibrated_target`:

- `serendipity`: `+0.00609`
- `cultural_calibration_kl`: `-0.00848`
- `minority_exposure_at_k`: `+0.03917`
- `target_culture_prob_mean`: `+0.00199`

Comparison file:

- `comparisons/pal_ot_cal_p3_balanced_vs_stage3_target.md`

This is the most balanced point and is the safest default candidate for paper reporting.

### `pal_ot_cal_p5_target_minor`

Compared with the current reference `dcas_full_ot_calibrated_target`:

- `serendipity`: `+0.00309`
- `cultural_calibration_kl`: `-0.00983`
- `minority_exposure_at_k`: `+0.06158`
- `target_culture_prob_mean`: `+0.00241`

Comparison file:

- `comparisons/pal_ot_cal_p5_target_minor_vs_stage3_target.md`

This point is slightly more aggressive on minority exposure while still keeping a positive serendipity gain.

## Takeaway

The migrated real PAL checkpoint is useful. The earlier mixed result came from evaluating it only at the old default target-calibrated point.

Once rerank weights are re-tuned for the PAL checkpoint itself, the model can surpass the current stage3 target reference on all four core metrics.
