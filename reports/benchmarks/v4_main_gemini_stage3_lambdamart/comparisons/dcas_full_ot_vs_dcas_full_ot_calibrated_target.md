# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `E:\Desktop\Echo\reports\benchmarks\v4_main_gemini_stage3_lambdamart\eval\dcas_full_ot.json`
- candidate: `E:\Desktop\Echo\reports\benchmarks\v4_main_gemini_stage3_lambdamart\eval\dcas_full_ot_calibrated_target.json`
- paired samples: `2400`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| serendipity | 0.8542503419 | 0.8244925563 | -0.0297577855 | [-0.032223, -0.027530] | 0.004975 | -0.492177 |
| cultural_calibration_kl | 2.3250431320 | 2.3104056591 | -0.0146374729 | [-0.015510, -0.013901] | 0.004975 | -0.686344 |
| minority_exposure_at_k | 0.1959791667 | 0.3759791667 | +0.1800000000 | [0.176187, 0.184090] | 0.004975 | 1.706208 |

