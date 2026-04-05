# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `E:\Desktop\Echo\reports\hparam\v4_routeA_small_gemini_stage3_calibration_sweep\eval\dcas_ot_cal_p3_balanced.json`
- candidate: `E:\Desktop\Echo\reports\hparam\v4_routeA_small_gemini_stage3_calibration_sweep\eval\dcas_ot_cal_p2_target.json`
- paired samples: `384`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| serendipity | 0.8619872645 | 0.8641977006 | +0.0022104361 | [0.000472, 0.004206] | 0.024876 | 0.115928 |
| cultural_calibration_kl | 1.5534537155 | 1.5501961971 | -0.0032575183 | [-0.003567, -0.002881] | 0.004975 | -0.867978 |
| minority_exposure_at_k | 0.5686197917 | 0.4997395833 | -0.0688802083 | [-0.074229, -0.063271] | 0.004975 | -1.361269 |

