# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `E:\Desktop\Echo\reports\benchmarks\v4_routeA_small_culturemert_lambdamart\eval\bpr_two_stage_hybrid.json`
- candidate: `E:\Desktop\Echo\reports\benchmarks\v4_routeA_small_culturemert_lambdamart\eval\dcas_full_ot_calibrated_target.json`
- paired samples: `384`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| serendipity | 0.4997631752 | 0.8468967773 | +0.3471336020 | [0.328627, 0.366196] | 0.004975 | 2.083639 |
| cultural_calibration_kl | 1.1064923571 | 1.0872535167 | -0.0192388404 | [-0.026406, -0.012531] | 0.004975 | -0.279224 |
| minority_exposure_at_k | 0.1940104167 | 0.4427083333 | +0.2486979167 | [0.234105, 0.262285] | 0.004975 | 1.882530 |

