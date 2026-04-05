# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `E:\Desktop\Echo\reports\hparam\v4_main_culturemert_stage3_calibration_sweep\eval\dcas_ot_cal_p3_balanced.json`
- candidate: `E:\Desktop\Echo\reports\hparam\v4_main_culturemert_stage3_calibration_sweep\eval\dcas_ot_cal_p2_target.json`
- paired samples: `2400`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| serendipity | 0.8298564343 | 0.8315641066 | +0.0017076723 | [0.000843, 0.002546] | 0.004975 | 0.076092 |
| cultural_calibration_kl | 2.0399801046 | 2.0296378083 | -0.0103422963 | [-0.011034, -0.009594] | 0.004975 | -0.578391 |
| minority_exposure_at_k | 0.4525416667 | 0.4023333333 | -0.0502083333 | [-0.052189, -0.048291] | 0.004975 | -1.097599 |

