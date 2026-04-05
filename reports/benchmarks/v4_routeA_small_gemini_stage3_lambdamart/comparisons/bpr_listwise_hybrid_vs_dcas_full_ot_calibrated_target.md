# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `E:\Desktop\Echo\reports\benchmarks\v4_routeA_small_gemini_stage3_lambdamart\eval\bpr_listwise_hybrid.json`
- candidate: `E:\Desktop\Echo\reports\benchmarks\v4_routeA_small_gemini_stage3_lambdamart\eval\dcas_full_ot_calibrated_target.json`
- paired samples: `384`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| serendipity | 0.7331212743 | 0.8641977006 | +0.1310764262 | [0.121259, 0.142093] | 0.000999 | 1.306958 |
| cultural_calibration_kl | 1.5662716439 | 1.5501961971 | -0.0160754468 | [-0.018178, -0.014035] | 0.000999 | -0.763597 |
| minority_exposure_at_k | 0.1656250000 | 0.4997395833 | +0.3341145833 | [0.317708, 0.352220] | 0.000999 | 1.917110 |

