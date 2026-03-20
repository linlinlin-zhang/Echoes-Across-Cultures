# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `E:\Desktop\Echo\reports\benchmarks\v4_routeA_small_gemini_stage3_lambdamart\eval\knn.json`
- candidate: `E:\Desktop\Echo\reports\benchmarks\v4_routeA_small_gemini_stage3_lambdamart\eval\dcas_full_ot_calibrated_target.json`
- paired samples: `384`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| serendipity | 0.7880963249 | 0.8641977006 | +0.0761013756 | [0.065677, 0.085903] | 0.004975 | 0.733102 |
| cultural_calibration_kl | 1.5801360410 | 1.5501961971 | -0.0299398439 | [-0.032377, -0.027517] | 0.004975 | -1.121762 |
| minority_exposure_at_k | 0.1842447917 | 0.4997395833 | +0.3154947917 | [0.296611, 0.332178] | 0.004975 | 1.853208 |

