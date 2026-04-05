# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `E:\Desktop\Echo\reports\hparam\v4_routeA_small_culturemert_stage3_calibration_sweep\eval\dcas_ot_cal_p3_balanced.json`
- candidate: `E:\Desktop\Echo\reports\hparam\v4_routeA_small_culturemert_stage3_calibration_sweep\eval\dcas_ot_cal_p2_target.json`
- paired samples: `384`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| serendipity | 0.8384035805 | 0.8406298893 | +0.0022263088 | [-0.000277, 0.004590] | 0.084577 | 0.090998 |
| cultural_calibration_kl | 1.0435748159 | 1.0212990236 | -0.0222757923 | [-0.024659, -0.019947] | 0.004975 | -0.971396 |
| minority_exposure_at_k | 0.5794270833 | 0.5095052083 | -0.0699218750 | [-0.074482, -0.065872] | 0.004975 | -1.574747 |

