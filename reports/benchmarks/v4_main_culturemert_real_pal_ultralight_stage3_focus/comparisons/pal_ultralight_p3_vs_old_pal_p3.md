# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `reports\hparam\v4_main_culturemert_real_pal_stage3_calibration_sweep\eval\pal_ot_cal_p3_balanced.json`
- candidate: `reports\benchmarks\v4_main_culturemert_real_pal_ultralight_stage3_focus\eval\pal_ultralight_ot_cal_p3_balanced.json`
- paired samples: `2400`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| serendipity | 0.8376541748 | 0.8366039213 | -0.0010502535 | [-0.003523, 0.001697] | 0.458472 | -0.014677 |
| cultural_calibration_kl | 2.0211533760 | 2.0325598734 | +0.0114064974 | [0.009757, 0.013441] | 0.003322 | 0.246034 |
| minority_exposure_at_k | 0.4415000000 | 0.4668125000 | +0.0253125000 | [0.020467, 0.030573] | 0.003322 | 0.195420 |
| target_culture_prob_mean | 0.1899555780 | 0.1856246550 | -0.0043309230 | [-0.004831, -0.003896] | 0.003322 | -0.369573 |

