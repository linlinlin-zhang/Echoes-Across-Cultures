# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `E:\Desktop\Echo\reports\benchmarks\v4_main_gemini_stage3_lambdamart\eval\bpr_lambdamart_hybrid.json`
- candidate: `E:\Desktop\Echo\reports\benchmarks\v4_main_gemini_stage3_lambdamart\eval\dcas_full_ot_calibrated_target.json`
- paired samples: `2400`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| serendipity | 0.7883640357 | 0.8244925563 | +0.0361285206 | [0.032845, 0.039829] | 0.000999 | 0.393841 |
| cultural_calibration_kl | 2.3182811905 | 2.3104056591 | -0.0078755313 | [-0.008575, -0.007218] | 0.000999 | -0.448620 |
| minority_exposure_at_k | 0.2746250000 | 0.3759791667 | +0.1013541667 | [0.094645, 0.108463] | 0.000999 | 0.590931 |

