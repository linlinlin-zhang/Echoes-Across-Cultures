# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `E:\Desktop\Echo\reports\benchmarks\v4_routeA_small_gemini_stage3_lambdamart\eval\dcas_full_ot_calibrated_minor.json`
- candidate: `E:\Desktop\Echo\reports\benchmarks\v4_routeA_small_gemini_stage3_lambdamart\eval\dcas_full_ot_calibrated_target.json`
- paired samples: `384`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| serendipity | 0.8585719854 | 0.8641977006 | +0.0056257152 | [0.002900, 0.008458] | 0.000999 | 0.202381 |
| cultural_calibration_kl | 1.5573534265 | 1.5501961971 | -0.0071572293 | [-0.007753, -0.006545] | 0.000999 | -1.211821 |
| minority_exposure_at_k | 0.6528645833 | 0.4997395833 | -0.1531250000 | [-0.160547, -0.145443] | 0.000999 | -1.958491 |

