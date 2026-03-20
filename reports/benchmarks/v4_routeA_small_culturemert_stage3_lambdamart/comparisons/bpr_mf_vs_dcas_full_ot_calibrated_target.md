# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `E:\Desktop\Echo\reports\benchmarks\v4_routeA_small_culturemert_stage3_lambdamart\eval\bpr_mf.json`
- candidate: `E:\Desktop\Echo\reports\benchmarks\v4_routeA_small_culturemert_stage3_lambdamart\eval\dcas_full_ot_calibrated_target.json`
- paired samples: `384`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| serendipity | 0.4810933998 | 0.8406298893 | +0.3595364895 | [0.343444, 0.374721] | 0.004975 | 2.178987 |
| cultural_calibration_kl | 1.1292621574 | 1.0212990236 | -0.1079631338 | [-0.117542, -0.098900] | 0.004975 | -1.265867 |
| minority_exposure_at_k | 0.0878906250 | 0.5095052083 | +0.4216145833 | [0.409889, 0.432839] | 0.004975 | 3.655253 |

