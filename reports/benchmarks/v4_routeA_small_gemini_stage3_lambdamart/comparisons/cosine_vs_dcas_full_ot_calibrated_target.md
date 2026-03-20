# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `E:\Desktop\Echo\reports\benchmarks\v4_routeA_small_gemini_stage3_lambdamart\eval\cosine.json`
- candidate: `E:\Desktop\Echo\reports\benchmarks\v4_routeA_small_gemini_stage3_lambdamart\eval\dcas_full_ot_calibrated_target.json`
- paired samples: `384`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| serendipity | 0.7901289721 | 0.8641977006 | +0.0740687284 | [0.063456, 0.083670] | 0.004975 | 0.701211 |
| cultural_calibration_kl | 1.5817551246 | 1.5501961971 | -0.0315589275 | [-0.034039, -0.029172] | 0.004975 | -1.219012 |
| minority_exposure_at_k | 0.1914062500 | 0.4997395833 | +0.3083333333 | [0.290713, 0.325160] | 0.004975 | 1.801188 |

