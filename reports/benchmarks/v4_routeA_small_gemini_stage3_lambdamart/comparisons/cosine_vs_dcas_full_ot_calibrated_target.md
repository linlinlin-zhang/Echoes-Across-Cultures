# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `E:\Desktop\Echo\reports\benchmarks\v4_routeA_small_gemini_stage3_lambdamart\eval\cosine.json`
- candidate: `E:\Desktop\Echo\reports\benchmarks\v4_routeA_small_gemini_stage3_lambdamart\eval\dcas_full_ot_calibrated_target.json`
- paired samples: `384`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| serendipity | 0.7901289721 | 0.8641977006 | +0.0740687284 | [0.063336, 0.084309] | 0.000999 | 0.701211 |
| cultural_calibration_kl | 1.5817551246 | 1.5501961971 | -0.0315589275 | [-0.034027, -0.028896] | 0.000999 | -1.219012 |
| minority_exposure_at_k | 0.1914062500 | 0.4997395833 | +0.3083333333 | [0.291146, 0.325010] | 0.000999 | 1.801188 |

