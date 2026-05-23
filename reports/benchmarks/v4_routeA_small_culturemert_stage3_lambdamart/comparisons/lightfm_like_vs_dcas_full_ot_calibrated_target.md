# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `E:\Desktop\Echo\reports\benchmarks\v4_routeA_small_culturemert_stage3_lambdamart\eval\lightfm_like.json`
- candidate: `E:\Desktop\Echo\reports\benchmarks\v4_routeA_small_culturemert_stage3_lambdamart\eval\dcas_full_ot_calibrated_target.json`
- paired samples: `384`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| serendipity | 0.3846401312 | 0.8406298893 | +0.4559897582 | [0.436542, 0.473372] | 0.000999 | 2.501826 |
| cultural_calibration_kl | 1.1485513081 | 1.0212990236 | -0.1272522844 | [-0.138107, -0.115434] | 0.000999 | -1.123229 |
| minority_exposure_at_k | 0.1194010417 | 0.5095052083 | +0.3901041667 | [0.376429, 0.404040] | 0.000999 | 2.999713 |

