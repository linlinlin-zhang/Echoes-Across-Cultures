# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `E:\Desktop\Echo\reports\benchmarks\v3_main_culturemert_stage3_lambdamart\eval\dcas_full_ot_calibrated_minor.json`
- candidate: `E:\Desktop\Echo\reports\benchmarks\v3_main_culturemert_stage3_lambdamart\eval\dcas_full_ot_calibrated_target.json`
- paired samples: `2400`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| serendipity | 0.8404395024 | 0.8385876200 | -0.0018518824 | [-0.003379, -0.000070] | 0.034826 | -0.046915 |
| cultural_calibration_kl | 1.9148020188 | 1.8792946220 | -0.0355073969 | [-0.038160, -0.032512] | 0.004975 | -0.496574 |
| minority_exposure_at_k | 0.5189791667 | 0.3814375000 | -0.1375416667 | [-0.141692, -0.133431] | 0.004975 | -1.319471 |

