# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `E:\Desktop\Echo\reports\benchmarks\v4_routeA_small_gemini_stage3_lambdamart\eval\popularity.json`
- candidate: `E:\Desktop\Echo\reports\benchmarks\v4_routeA_small_gemini_stage3_lambdamart\eval\dcas_full_ot_calibrated_target.json`
- paired samples: `384`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| serendipity | 0.6603375737 | 0.8641977006 | +0.2038601268 | [0.191627, 0.216957] | 0.000999 | 1.642065 |
| cultural_calibration_kl | 1.5805226462 | 1.5501961971 | -0.0303264490 | [-0.032201, -0.028402] | 0.000999 | -1.593593 |
| minority_exposure_at_k | 0.0000000000 | 0.4997395833 | +0.4997395833 | [0.485804, 0.514977] | 0.000999 | 3.499599 |

