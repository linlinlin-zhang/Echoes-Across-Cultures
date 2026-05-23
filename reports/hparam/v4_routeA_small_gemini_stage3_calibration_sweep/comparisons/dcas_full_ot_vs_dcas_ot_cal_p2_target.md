# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `E:\Desktop\Echo\reports\hparam\v4_routeA_small_gemini_stage3_calibration_sweep\eval\dcas_full_ot.json`
- candidate: `E:\Desktop\Echo\reports\hparam\v4_routeA_small_gemini_stage3_calibration_sweep\eval\dcas_ot_cal_p2_target.json`
- paired samples: `384`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| serendipity | 0.8605679910 | 0.8641977006 | +0.0036297096 | [-0.002518, 0.009982] | 0.248756 | 0.064086 |
| cultural_calibration_kl | 1.5723317815 | 1.5501961971 | -0.0221355843 | [-0.023386, -0.021082] | 0.004975 | -1.787241 |
| minority_exposure_at_k | 0.2423177083 | 0.4997395833 | +0.2574218750 | [0.245433, 0.269668] | 0.004975 | 2.062484 |

