# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `E:\Desktop\Echo\reports\benchmarks\v4_routeA_small_gemini_stage3_lambdamart\eval\dcas_full_ot.json`
- candidate: `E:\Desktop\Echo\reports\benchmarks\v4_routeA_small_gemini_stage3_lambdamart\eval\dcas_full_ot_calibrated_target.json`
- paired samples: `384`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| serendipity | 0.8605679910 | 0.8641977006 | +0.0036297096 | [-0.002075, 0.009525] | 0.208791 | 0.064086 |
| cultural_calibration_kl | 1.5723317815 | 1.5501961971 | -0.0221355843 | [-0.023485, -0.020888] | 0.000999 | -1.787241 |
| minority_exposure_at_k | 0.2423177083 | 0.4997395833 | +0.2574218750 | [0.245703, 0.270189] | 0.000999 | 2.062484 |

