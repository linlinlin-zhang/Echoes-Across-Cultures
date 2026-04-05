# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `E:\Desktop\Echo\reports\hparam\v4_routeA_small_culturemert_stage3_calibration_sweep\eval\dcas_ot_cal_p4_minor.json`
- candidate: `E:\Desktop\Echo\reports\hparam\v4_routeA_small_culturemert_stage3_calibration_sweep\eval\dcas_ot_cal_p2_target.json`
- paired samples: `384`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| serendipity | 0.8370797699 | 0.8406298893 | +0.0035501194 | [-0.000476, 0.007951] | 0.079602 | 0.095518 |
| cultural_calibration_kl | 1.0747450574 | 1.0212990236 | -0.0534460338 | [-0.058654, -0.048801] | 0.004975 | -1.289632 |
| minority_exposure_at_k | 0.6798177083 | 0.5095052083 | -0.1703125000 | [-0.175527, -0.163773] | 0.004975 | -2.766483 |

