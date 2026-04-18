# V4 Main CultureMERT Real PAL Ultralight Stage3 Focus Benchmark

## Purpose

This benchmark evaluates whether an additional light real-PAL fine-tune can beat the previously best migrated PAL checkpoint under the main benchmark definition.

Config:

- `configs/benchmark/recommender_benchmark_v4_main_culturemert_real_pal_ultralight_stage3_focus.run.json`

Checkpoint:

- `storage/models/pal/v4_main_culturemert_real_from_v4_main_annotation_stage3_ultralight_p3/real_pal_model.pt`

## Main Result

The ultralight fine-tuned checkpoint is useful, but it is not the new best model.

Compared with the official reference `dcas_full_ot_calibrated_target`, `pal_ultralight_ot_cal_p3_balanced` improves:

- `serendipity`
- `minority_exposure_at_k`

But it worsens:

- `cultural_calibration_kl`
- `target_culture_prob_mean`

## Decision

The current best paper-ready PAL candidates remain:

- `pal_ot_cal_p3_balanced`
- `pal_ot_cal_p5_target_minor`

from:

- `reports/hparam/v4_main_culturemert_real_pal_stage3_calibration_sweep/`

The new ultralight checkpoint is better understood as an exploratory training-side follow-up, not as the new main result.
