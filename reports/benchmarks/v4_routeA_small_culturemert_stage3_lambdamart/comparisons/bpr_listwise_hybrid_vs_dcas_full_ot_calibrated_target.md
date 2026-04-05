# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `E:\Desktop\Echo\reports\benchmarks\v4_routeA_small_culturemert_stage3_lambdamart\eval\bpr_listwise_hybrid.json`
- candidate: `E:\Desktop\Echo\reports\benchmarks\v4_routeA_small_culturemert_stage3_lambdamart\eval\dcas_full_ot_calibrated_target.json`
- paired samples: `384`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| serendipity | 0.5041031100 | 0.8406298893 | +0.3365267793 | [0.319263, 0.353760] | 0.000999 | 1.979128 |
| cultural_calibration_kl | 1.1199350047 | 1.0212990236 | -0.0986359811 | [-0.105862, -0.091024] | 0.000999 | -1.318976 |
| minority_exposure_at_k | 0.1921875000 | 0.5095052083 | +0.3173177083 | [0.304033, 0.330990] | 0.000999 | 2.317992 |

