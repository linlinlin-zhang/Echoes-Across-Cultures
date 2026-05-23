# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `E:\Desktop\Echo\reports\benchmarks\v4_routeA_small_gemini_stage3_lambdamart\eval\lightfm_like.json`
- candidate: `E:\Desktop\Echo\reports\benchmarks\v4_routeA_small_gemini_stage3_lambdamart\eval\dcas_full_ot_calibrated_target.json`
- paired samples: `384`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| serendipity | 0.6844185521 | 0.8641977006 | +0.1797791485 | [0.167597, 0.191697] | 0.000999 | 1.584040 |
| cultural_calibration_kl | 1.5773079058 | 1.5501961971 | -0.0271117086 | [-0.029392, -0.024701] | 0.000999 | -1.202138 |
| minority_exposure_at_k | 0.0460937500 | 0.4997395833 | +0.4536458333 | [0.439193, 0.469271] | 0.000999 | 2.963477 |

