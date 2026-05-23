# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `reports\benchmarks\v4_main_culturemert_stage3_lambdamart\eval\dcas_full_ot_calibrated_target.json`
- candidate: `reports\hparam\v4_main_culturemert_real_pal_stage3_calibration_sweep\eval\pal_ot_cal_p3_balanced.json`
- paired samples: `2400`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| serendipity | 0.8315641066 | 0.8376541748 | +0.0060900682 | [0.003789, 0.008605] | 0.003322 | 0.097379 |
| cultural_calibration_kl | 2.0296378083 | 2.0211533760 | -0.0084844323 | [-0.010663, -0.006465] | 0.003322 | -0.164544 |
| minority_exposure_at_k | 0.4023333333 | 0.4415000000 | +0.0391666667 | [0.033791, 0.044078] | 0.003322 | 0.290351 |
| target_culture_prob_mean | 0.1879636552 | 0.1899555780 | +0.0019919228 | [0.001430, 0.002483] | 0.003322 | 0.165015 |

