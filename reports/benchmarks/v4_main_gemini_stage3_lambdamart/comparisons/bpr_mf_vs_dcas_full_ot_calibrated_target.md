# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `E:\Desktop\Echo\reports\benchmarks\v4_main_gemini_stage3_lambdamart\eval\bpr_mf.json`
- candidate: `E:\Desktop\Echo\reports\benchmarks\v4_main_gemini_stage3_lambdamart\eval\dcas_full_ot_calibrated_target.json`
- paired samples: `2400`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| serendipity | 0.7747378074 | 0.8244925563 | +0.0497547489 | [0.045123, 0.053696] | 0.004975 | 0.517875 |
| cultural_calibration_kl | 2.3214217623 | 2.3104056591 | -0.0110161032 | [-0.011957, -0.010217] | 0.004975 | -0.483988 |
| minority_exposure_at_k | 0.1645208333 | 0.3759791667 | +0.2114583333 | [0.205644, 0.217604] | 0.004975 | 1.325500 |

