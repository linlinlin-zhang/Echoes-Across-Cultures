# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `E:\Desktop\Echo\reports\benchmarks\v4_routeA_small_culturemert_stage3_lambdamart\eval\dcas_full_ot_calibrated_minor.json`
- candidate: `E:\Desktop\Echo\reports\benchmarks\v4_routeA_small_culturemert_stage3_lambdamart\eval\dcas_full_ot_calibrated_target.json`
- paired samples: `384`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| serendipity | 0.8370797699 | 0.8406298893 | +0.0035501194 | [0.000101, 0.007141] | 0.067932 | 0.095518 |
| cultural_calibration_kl | 1.0747450574 | 1.0212990236 | -0.0534460338 | [-0.057660, -0.049510] | 0.000999 | -1.289632 |
| minority_exposure_at_k | 0.6798177083 | 0.5095052083 | -0.1703125000 | [-0.176302, -0.164193] | 0.000999 | -2.766483 |

