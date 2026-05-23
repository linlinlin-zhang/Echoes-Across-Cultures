# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `E:\Desktop\Echo\reports\hparam\v4_main_culturemert_real_pal_stage3_calibration_sweep\eval\pal_ot_cal_p5_target_minor.json`
- candidate: `E:\Desktop\Echo\reports\hparam\v4_main_culturemert_real_pal_stage3_calibration_sweep\eval\pal_ot_cal_p2_target.json`
- paired samples: `2400`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| serendipity | 0.8346559840 | 0.8369521178 | +0.0022961338 | [0.001406, 0.003349] | 0.004975 | 0.091415 |
| cultural_calibration_kl | 2.0198058234 | 2.0196814106 | -0.0001244129 | [-0.000750, 0.000645] | 0.751244 | -0.007300 |
| minority_exposure_at_k | 0.4639166667 | 0.3865625000 | -0.0773541667 | [-0.079418, -0.075019] | 0.004975 | -1.315761 |

