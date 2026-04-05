# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `E:\Desktop\Echo\reports\benchmarks\v4_main_gemini_stage3_lambdamart\eval\dcas_full_ot_calibrated_minor.json`
- candidate: `E:\Desktop\Echo\reports\benchmarks\v4_main_gemini_stage3_lambdamart\eval\dcas_full_ot_calibrated_target.json`
- paired samples: `2400`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| serendipity | 0.8208598428 | 0.8244925563 | +0.0036327135 | [0.002305, 0.004903] | 0.000999 | 0.110601 |
| cultural_calibration_kl | 2.3129104862 | 2.3104056591 | -0.0025048271 | [-0.002612, -0.002404] | 0.000999 | -0.968857 |
| minority_exposure_at_k | 0.4799583333 | 0.3759791667 | -0.1039791667 | [-0.106917, -0.101080] | 0.000999 | -1.485455 |

