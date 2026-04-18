# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `E:\Desktop\Echo\reports\benchmarks\v4_main_culturemert_real_pal_ultralight_stage3_focus\eval\pal_ultralight_ot.json`
- candidate: `E:\Desktop\Echo\reports\benchmarks\v4_main_culturemert_real_pal_ultralight_stage3_focus\eval\pal_ultralight_ot_cal_p3_balanced.json`
- paired samples: `2400`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| serendipity | 0.8551924237 | 0.8366039213 | -0.0185885024 | [-0.020874, -0.016408] | 0.004975 | -0.371124 |
| cultural_calibration_kl | 2.0942911023 | 2.0325598734 | -0.0617312289 | [-0.065698, -0.058440] | 0.004975 | -0.625921 |
| minority_exposure_at_k | 0.2442083333 | 0.4668125000 | +0.2226041667 | [0.218873, 0.226335] | 0.004975 | 2.146279 |

