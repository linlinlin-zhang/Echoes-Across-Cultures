# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `E:\Desktop\Echo\reports\benchmarks\v4_main_culturemert_stage3_lambdamart\eval\dcas_full_ot.json`
- candidate: `E:\Desktop\Echo\reports\benchmarks\v4_main_culturemert_stage3_lambdamart\eval\dcas_full_ot_calibrated_target.json`
- paired samples: `2400`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| serendipity | 0.8578608208 | 0.8315641066 | -0.0262967142 | [-0.028465, -0.024676] | 0.004975 | -0.519174 |
| cultural_calibration_kl | 2.0825807375 | 2.0296378083 | -0.0529429292 | [-0.057640, -0.048171] | 0.004975 | -0.456378 |
| minority_exposure_at_k | 0.2460208333 | 0.4023333333 | +0.1563125000 | [0.152866, 0.159731] | 0.004975 | 1.730584 |

