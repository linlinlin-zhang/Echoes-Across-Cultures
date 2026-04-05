# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `E:\Desktop\Echo\reports\hparam\v4_routeA_small_gemini_stage3_calibration_sweep\eval\dcas_ot_cal_p4_minor.json`
- candidate: `E:\Desktop\Echo\reports\hparam\v4_routeA_small_gemini_stage3_calibration_sweep\eval\dcas_ot_cal_p2_target.json`
- paired samples: `384`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| serendipity | 0.8585719854 | 0.8641977006 | +0.0056257152 | [0.003077, 0.008453] | 0.004975 | 0.202381 |
| cultural_calibration_kl | 1.5573534265 | 1.5501961971 | -0.0071572293 | [-0.007714, -0.006530] | 0.004975 | -1.211821 |
| minority_exposure_at_k | 0.6528645833 | 0.4997395833 | -0.1531250000 | [-0.160547, -0.145309] | 0.004975 | -1.958491 |

