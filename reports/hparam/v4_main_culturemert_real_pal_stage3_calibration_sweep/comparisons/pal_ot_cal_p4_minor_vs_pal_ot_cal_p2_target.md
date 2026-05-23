# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `E:\Desktop\Echo\reports\hparam\v4_main_culturemert_real_pal_stage3_calibration_sweep\eval\pal_ot_cal_p4_minor.json`
- candidate: `E:\Desktop\Echo\reports\hparam\v4_main_culturemert_real_pal_stage3_calibration_sweep\eval\pal_ot_cal_p2_target.json`
- paired samples: `2400`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| serendipity | 0.8343536393 | 0.8369521178 | +0.0025984785 | [0.001507, 0.003700] | 0.004975 | 0.091287 |
| cultural_calibration_kl | 2.0318671791 | 2.0196814106 | -0.0121857685 | [-0.013004, -0.011382] | 0.004975 | -0.601736 |
| minority_exposure_at_k | 0.5077083333 | 0.3865625000 | -0.1211458333 | [-0.124045, -0.118208] | 0.004975 | -1.757838 |

