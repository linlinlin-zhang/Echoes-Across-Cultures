# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `E:\Desktop\Echo\reports\benchmarks\v4_main_culturemert_stage3_lambdamart\eval\popularity.json`
- candidate: `E:\Desktop\Echo\reports\benchmarks\v4_main_culturemert_stage3_lambdamart\eval\dcas_full_ot_calibrated_target.json`
- paired samples: `2400`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| serendipity | 0.5014640337 | 0.8315641066 | +0.3301000729 | [0.322620, 0.336895] | 0.000999 | 1.819250 |
| cultural_calibration_kl | 2.1734106388 | 2.0296378083 | -0.1437728305 | [-0.154772, -0.132923] | 0.000999 | -0.526163 |
| minority_exposure_at_k | 0.0000000000 | 0.4023333333 | +0.4023333333 | [0.395770, 0.408668] | 0.000999 | 2.386615 |

