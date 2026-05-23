# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `E:\Desktop\Echo\reports\benchmarks\v4_routeA_small_culturemert_stage3_lambdamart\eval\popularity.json`
- candidate: `E:\Desktop\Echo\reports\benchmarks\v4_routeA_small_culturemert_stage3_lambdamart\eval\dcas_full_ot_calibrated_target.json`
- paired samples: `384`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| serendipity | 0.3865871045 | 0.8406298893 | +0.4540427848 | [0.432907, 0.474495] | 0.000999 | 2.156148 |
| cultural_calibration_kl | 1.1207042194 | 1.0212990236 | -0.0994051958 | [-0.109260, -0.089857] | 0.000999 | -1.003762 |
| minority_exposure_at_k | 0.0000000000 | 0.5095052083 | +0.5095052083 | [0.500260, 0.519141] | 0.000999 | 5.624569 |

