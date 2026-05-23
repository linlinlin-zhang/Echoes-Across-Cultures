# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `E:\Desktop\Echo\reports\hparam\v4_routeA_small_culturemert_stage3_calibration_sweep\eval\dcas_ot_cal_p5_ultra_minor.json`
- candidate: `E:\Desktop\Echo\reports\hparam\v4_routeA_small_culturemert_stage3_calibration_sweep\eval\dcas_ot_cal_p2_target.json`
- paired samples: `384`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| serendipity | 0.8384543236 | 0.8406298893 | +0.0021755657 | [-0.001869, 0.006625] | 0.328358 | 0.053876 |
| cultural_calibration_kl | 1.0895369249 | 1.0212990236 | -0.0682379013 | [-0.075067, -0.062409] | 0.004975 | -1.333509 |
| minority_exposure_at_k | 0.7528645833 | 0.5095052083 | -0.2433593750 | [-0.249232, -0.236973] | 0.004975 | -3.907522 |

