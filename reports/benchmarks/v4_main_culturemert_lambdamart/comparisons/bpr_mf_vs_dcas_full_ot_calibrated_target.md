# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `E:\Desktop\Echo\reports\benchmarks\v4_main_culturemert_lambdamart\eval\bpr_mf.json`
- candidate: `E:\Desktop\Echo\reports\benchmarks\v4_main_culturemert_lambdamart\eval\dcas_full_ot_calibrated_target.json`
- paired samples: `2400`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| serendipity | 0.5372616786 | 0.8110557968 | +0.2737941182 | [0.265845, 0.279542] | 0.004975 | 1.676846 |
| cultural_calibration_kl | 2.1138557933 | 2.0492074270 | -0.0646483663 | [-0.071429, -0.058492] | 0.004975 | -0.404096 |
| minority_exposure_at_k | 0.1615416667 | 0.3998125000 | +0.2382708333 | [0.231520, 0.245417] | 0.004975 | 1.460658 |

