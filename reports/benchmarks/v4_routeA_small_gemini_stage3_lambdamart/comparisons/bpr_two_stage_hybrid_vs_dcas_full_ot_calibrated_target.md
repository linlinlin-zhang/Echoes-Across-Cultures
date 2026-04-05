# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `E:\Desktop\Echo\reports\benchmarks\v4_routeA_small_gemini_stage3_lambdamart\eval\bpr_two_stage_hybrid.json`
- candidate: `E:\Desktop\Echo\reports\benchmarks\v4_routeA_small_gemini_stage3_lambdamart\eval\dcas_full_ot_calibrated_target.json`
- paired samples: `384`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| serendipity | 0.7279575329 | 0.8641977006 | +0.1362401676 | [0.125914, 0.146936] | 0.000999 | 1.360918 |
| cultural_calibration_kl | 1.5684985653 | 1.5501961971 | -0.0183023681 | [-0.020230, -0.016407] | 0.000999 | -0.954646 |
| minority_exposure_at_k | 0.1451822917 | 0.4997395833 | +0.3545572917 | [0.338278, 0.372142] | 0.000999 | 2.083202 |

