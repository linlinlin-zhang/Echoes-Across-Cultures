# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `E:\Desktop\Echo\reports\benchmarks\v4_routeA_small_gemini_stage3_lambdamart\eval\bpr_lambdamart_hybrid.json`
- candidate: `E:\Desktop\Echo\reports\benchmarks\v4_routeA_small_gemini_stage3_lambdamart\eval\dcas_full_ot_calibrated_target.json`
- paired samples: `384`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| serendipity | 0.7266650412 | 0.8641977006 | +0.1375326593 | [0.127225, 0.148417] | 0.000999 | 1.354270 |
| cultural_calibration_kl | 1.5676303241 | 1.5501961971 | -0.0174341270 | [-0.019375, -0.015456] | 0.000999 | -0.896369 |
| minority_exposure_at_k | 0.1513020833 | 0.4997395833 | +0.3484375000 | [0.331374, 0.368099] | 0.000999 | 1.961227 |

