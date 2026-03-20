# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `E:\Desktop\Echo\reports\benchmarks\v4_routeA_small_culturemert_stage3_lambdamart\eval\bpr_listwise_hybrid.json`
- candidate: `E:\Desktop\Echo\reports\benchmarks\v4_routeA_small_culturemert_stage3_lambdamart\eval\dcas_full_ot_calibrated_target.json`
- paired samples: `384`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| serendipity | 0.5041031100 | 0.8406298893 | +0.3365267793 | [0.320193, 0.351542] | 0.004975 | 1.979128 |
| cultural_calibration_kl | 1.1199350047 | 1.0212990236 | -0.0986359811 | [-0.106973, -0.090845] | 0.004975 | -1.318976 |
| minority_exposure_at_k | 0.1921875000 | 0.5095052083 | +0.3173177083 | [0.304544, 0.330736] | 0.004975 | 2.317992 |

