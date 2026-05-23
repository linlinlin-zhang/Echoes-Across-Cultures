# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `reports\hparam\v4_main_culturemert_real_pal_stage3_calibration_sweep\eval\pal_ot_cal_p5_target_minor.json`
- candidate: `reports\benchmarks\v4_main_culturemert_real_pal_ultralight_stage3_focus\eval\pal_ultralight_ot_cal_p5_target_minor.json`
- paired samples: `2400`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| serendipity | 0.8346559840 | 0.8370190908 | +0.0023631068 | [-0.000437, 0.005256] | 0.099668 | 0.032165 |
| cultural_calibration_kl | 2.0198058234 | 2.0303214609 | +0.0105156375 | [0.008832, 0.012622] | 0.003322 | 0.216655 |
| minority_exposure_at_k | 0.4639166667 | 0.4818125000 | +0.0178958333 | [0.012781, 0.023897] | 0.003322 | 0.132234 |
| target_culture_prob_mean | 0.1903699362 | 0.1861274222 | -0.0042425140 | [-0.004775, -0.003783] | 0.003322 | -0.350230 |

