# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `E:\Desktop\Echo\reports\benchmarks\v4_main_culturemert_stage3_lambdamart\eval\knn.json`
- candidate: `E:\Desktop\Echo\reports\benchmarks\v4_main_culturemert_stage3_lambdamart\eval\dcas_full_ot_calibrated_target.json`
- paired samples: `2400`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| serendipity | 0.6445218086 | 0.8315641066 | +0.1870422980 | [0.179500, 0.194299] | 0.000999 | 0.959322 |
| cultural_calibration_kl | 2.2341304751 | 2.0296378083 | -0.2044926669 | [-0.216228, -0.193497] | 0.000999 | -0.703022 |
| minority_exposure_at_k | 0.2130625000 | 0.4023333333 | +0.1892708333 | [0.184729, 0.194314] | 0.000999 | 1.574738 |

