# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `E:\Desktop\Echo\reports\benchmarks\v4_routeA_small_culturemert_stage3_lambdamart\eval\knn.json`
- candidate: `E:\Desktop\Echo\reports\benchmarks\v4_routeA_small_culturemert_stage3_lambdamart\eval\dcas_full_ot_calibrated_target.json`
- paired samples: `384`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| serendipity | 0.4451817214 | 0.8406298893 | +0.3954481679 | [0.372875, 0.415878] | 0.000999 | 1.816325 |
| cultural_calibration_kl | 1.1741739532 | 1.0212990236 | -0.1528749295 | [-0.165758, -0.140398] | 0.000999 | -1.200783 |
| minority_exposure_at_k | 0.2424479167 | 0.5095052083 | +0.2670572917 | [0.258073, 0.276826] | 0.000999 | 2.799330 |

