# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `E:\Desktop\Echo\reports\benchmarks\v4_main_gemini_stage3_lambdamart\eval\lightfm_like.json`
- candidate: `E:\Desktop\Echo\reports\benchmarks\v4_main_gemini_stage3_lambdamart\eval\dcas_full_ot_calibrated_target.json`
- paired samples: `2400`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| serendipity | 0.7666224697 | 0.8244925563 | +0.0578700866 | [0.054011, 0.061804] | 0.000999 | 0.595451 |
| cultural_calibration_kl | 2.3337010077 | 2.3104056591 | -0.0232953486 | [-0.024591, -0.022031] | 0.000999 | -0.707489 |
| minority_exposure_at_k | 0.1161875000 | 0.3759791667 | +0.2597916667 | [0.254021, 0.265708] | 0.000999 | 1.731345 |

