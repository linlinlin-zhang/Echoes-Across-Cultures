# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `E:\Desktop\Echo\reports\hparam\v4_main_culturemert_real_pal_stage3_calibration_sweep\eval\pal_ot.json`
- candidate: `E:\Desktop\Echo\reports\hparam\v4_main_culturemert_real_pal_stage3_calibration_sweep\eval\pal_ot_cal_p2_target.json`
- paired samples: `2400`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| serendipity | 0.8596242652 | 0.8369521178 | -0.0226721474 | [-0.025039, -0.020436] | 0.004975 | -0.359526 |
| cultural_calibration_kl | 2.0595241428 | 2.0196814106 | -0.0398427322 | [-0.044192, -0.036364] | 0.004975 | -0.421762 |
| minority_exposure_at_k | 0.2453125000 | 0.3865625000 | +0.1412500000 | [0.137625, 0.145043] | 0.004975 | 1.542517 |

