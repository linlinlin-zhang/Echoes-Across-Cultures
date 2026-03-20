# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `E:\Desktop\Echo\reports\benchmarks\v4_main_gemini_stage3_lambdamart\eval\bpr_listwise_hybrid.json`
- candidate: `E:\Desktop\Echo\reports\benchmarks\v4_main_gemini_stage3_lambdamart\eval\dcas_full_ot_calibrated_target.json`
- paired samples: `2400`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| serendipity | 0.7922040535 | 0.8244925563 | +0.0322885028 | [0.027756, 0.035430] | 0.004975 | 0.352077 |
| cultural_calibration_kl | 2.3178517884 | 2.3104056591 | -0.0074461293 | [-0.008206, -0.006682] | 0.004975 | -0.442978 |
| minority_exposure_at_k | 0.2868125000 | 0.3759791667 | +0.0891666667 | [0.082916, 0.096469] | 0.004975 | 0.518991 |

