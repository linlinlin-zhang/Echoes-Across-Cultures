# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `E:\Desktop\Echo\reports\hparam\v4_main_culturemert_real_pal_stage3_calibration_sweep\eval\pal_ot_cal_p3_balanced.json`
- candidate: `E:\Desktop\Echo\reports\hparam\v4_main_culturemert_real_pal_stage3_calibration_sweep\eval\pal_ot_cal_p2_target.json`
- paired samples: `2400`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| serendipity | 0.8376541748 | 0.8369521178 | -0.0007020570 | [-0.001562, 0.000186] | 0.208955 | -0.028505 |
| cultural_calibration_kl | 2.0211533760 | 2.0196814106 | -0.0014719654 | [-0.002038, -0.000922] | 0.004975 | -0.096251 |
| minority_exposure_at_k | 0.4415000000 | 0.3865625000 | -0.0549375000 | [-0.057088, -0.052727] | 0.004975 | -1.159663 |

