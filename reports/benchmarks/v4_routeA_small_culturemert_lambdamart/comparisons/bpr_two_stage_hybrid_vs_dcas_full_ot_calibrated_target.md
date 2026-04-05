# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `E:\Desktop\Echo\reports\benchmarks\v4_routeA_small_culturemert_lambdamart\eval\bpr_two_stage_hybrid.json`
- candidate: `E:\Desktop\Echo\reports\benchmarks\v4_routeA_small_culturemert_lambdamart\eval\dcas_full_ot_calibrated_target.json`
- paired samples: `384`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| serendipity | 0.4997631752 | 0.8468967773 | +0.3471336020 | [0.330686, 0.363661] | 0.000999 | 2.083639 |
| cultural_calibration_kl | 1.1064923571 | 1.0872535167 | -0.0192388404 | [-0.026148, -0.012074] | 0.000999 | -0.279224 |
| minority_exposure_at_k | 0.1940104167 | 0.4427083333 | +0.2486979167 | [0.235671, 0.263024] | 0.000999 | 1.882530 |

