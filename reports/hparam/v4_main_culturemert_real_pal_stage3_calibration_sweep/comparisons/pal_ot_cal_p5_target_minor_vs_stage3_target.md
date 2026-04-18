# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `reports\benchmarks\v4_main_culturemert_stage3_lambdamart\eval\dcas_full_ot_calibrated_target.json`
- candidate: `reports\hparam\v4_main_culturemert_real_pal_stage3_calibration_sweep\eval\pal_ot_cal_p5_target_minor.json`
- paired samples: `2400`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| serendipity | 0.8315641066 | 0.8346559840 | +0.0030918774 | [0.000687, 0.005712] | 0.023256 | 0.047660 |
| cultural_calibration_kl | 2.0296378083 | 2.0198058234 | -0.0098319848 | [-0.012052, -0.007782] | 0.003322 | -0.185172 |
| minority_exposure_at_k | 0.4023333333 | 0.4639166667 | +0.0615833333 | [0.055556, 0.066647] | 0.003322 | 0.426092 |
| target_culture_prob_mean | 0.1879636552 | 0.1903699362 | +0.0024062810 | [0.001923, 0.002882] | 0.003322 | 0.197609 |

