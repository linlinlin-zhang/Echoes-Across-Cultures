# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `E:\Desktop\Echo\reports\benchmarks\v4_main_gemini_stage3_lambdamart\eval\knn.json`
- candidate: `E:\Desktop\Echo\reports\benchmarks\v4_main_gemini_stage3_lambdamart\eval\dcas_full_ot_calibrated_target.json`
- paired samples: `2400`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| serendipity | 0.8542488298 | 0.8244925563 | -0.0297562735 | [-0.033305, -0.026248] | 0.000999 | -0.312288 |
| cultural_calibration_kl | 2.3337626527 | 2.3104056591 | -0.0233569936 | [-0.024511, -0.022236] | 0.000999 | -0.775531 |
| minority_exposure_at_k | 0.2245625000 | 0.3759791667 | +0.1514166667 | [0.145999, 0.157792] | 0.000999 | 1.007369 |

