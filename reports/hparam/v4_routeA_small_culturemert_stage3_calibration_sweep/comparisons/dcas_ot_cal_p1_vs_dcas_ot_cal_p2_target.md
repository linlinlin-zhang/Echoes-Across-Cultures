# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `E:\Desktop\Echo\reports\hparam\v4_routeA_small_culturemert_stage3_calibration_sweep\eval\dcas_ot_cal_p1.json`
- candidate: `E:\Desktop\Echo\reports\hparam\v4_routeA_small_culturemert_stage3_calibration_sweep\eval\dcas_ot_cal_p2_target.json`
- paired samples: `384`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| serendipity | 0.8377972828 | 0.8406298893 | +0.0028326066 | [0.000321, 0.005235] | 0.019900 | 0.123938 |
| cultural_calibration_kl | 1.0136133471 | 1.0212990236 | +0.0076856765 | [0.005752, 0.009587] | 0.004975 | 0.428355 |
| minority_exposure_at_k | 0.4373697917 | 0.5095052083 | +0.0721354167 | [0.067438, 0.076566] | 0.004975 | 1.543208 |

