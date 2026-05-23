# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `E:\Desktop\Echo\reports\hparam\v4_main_culturemert_real_pal_stage3_calibration_sweep\eval\pal_ot_cal_p1.json`
- candidate: `E:\Desktop\Echo\reports\hparam\v4_main_culturemert_real_pal_stage3_calibration_sweep\eval\pal_ot_cal_p2_target.json`
- paired samples: `2400`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| serendipity | 0.8410781933 | 0.8369521178 | -0.0041260755 | [-0.004997, -0.003092] | 0.004975 | -0.156320 |
| cultural_calibration_kl | 2.0137831355 | 2.0196814106 | +0.0058982750 | [0.005229, 0.006354] | 0.004975 | 0.393295 |
| minority_exposure_at_k | 0.3348333333 | 0.3865625000 | +0.0517291667 | [0.050082, 0.053418] | 0.004975 | 1.207837 |

