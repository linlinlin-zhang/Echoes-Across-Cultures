# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `E:\Desktop\Echo\reports\benchmarks\v4_main_culturemert_lambdamart\eval\bpr_lambdamart_hybrid.json`
- candidate: `E:\Desktop\Echo\reports\benchmarks\v4_main_culturemert_lambdamart\eval\dcas_full_ot_calibrated_target.json`
- paired samples: `2400`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| serendipity | 0.5593675914 | 0.8110557968 | +0.2516882053 | [0.243981, 0.257554] | 0.004975 | 1.569789 |
| cultural_calibration_kl | 2.0936865748 | 2.0492074270 | -0.0444791478 | [-0.050159, -0.040118] | 0.004975 | -0.356923 |
| minority_exposure_at_k | 0.2723125000 | 0.3998125000 | +0.1275000000 | [0.121179, 0.134697] | 0.004975 | 0.735458 |

