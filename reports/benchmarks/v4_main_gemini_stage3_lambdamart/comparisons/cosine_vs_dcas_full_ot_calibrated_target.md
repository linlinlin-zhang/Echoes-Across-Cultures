# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `E:\Desktop\Echo\reports\benchmarks\v4_main_gemini_stage3_lambdamart\eval\cosine.json`
- candidate: `E:\Desktop\Echo\reports\benchmarks\v4_main_gemini_stage3_lambdamart\eval\dcas_full_ot_calibrated_target.json`
- paired samples: `2400`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| serendipity | 0.8517191970 | 0.8244925563 | -0.0272266407 | [-0.031738, -0.023179] | 0.004975 | -0.283346 |
| cultural_calibration_kl | 2.3333918902 | 2.3104056591 | -0.0229862311 | [-0.024012, -0.022009] | 0.004975 | -0.834077 |
| minority_exposure_at_k | 0.2304791667 | 0.3759791667 | +0.1455000000 | [0.138810, 0.151752] | 0.004975 | 0.966302 |

