# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `E:\Desktop\Echo\reports\benchmarks\v4_main_culturemert_lambdamart\eval\bpr_mf.json`
- candidate: `E:\Desktop\Echo\reports\benchmarks\v4_main_culturemert_lambdamart\eval\dcas_full_ot_calibrated_target.json`
- paired samples: `2400`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| serendipity | 0.5372616786 | 0.8110557968 | +0.2737941182 | [0.267324, 0.280195] | 0.000999 | 1.676846 |
| cultural_calibration_kl | 2.1138557933 | 2.0492074270 | -0.0646483663 | [-0.071405, -0.058818] | 0.000999 | -0.404096 |
| minority_exposure_at_k | 0.1615416667 | 0.3998125000 | +0.2382708333 | [0.232166, 0.244813] | 0.000999 | 1.460658 |

