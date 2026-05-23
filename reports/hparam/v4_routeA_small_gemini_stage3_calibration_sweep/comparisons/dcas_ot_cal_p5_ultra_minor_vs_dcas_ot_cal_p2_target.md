# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `E:\Desktop\Echo\reports\hparam\v4_routeA_small_gemini_stage3_calibration_sweep\eval\dcas_ot_cal_p5_ultra_minor.json`
- candidate: `E:\Desktop\Echo\reports\hparam\v4_routeA_small_gemini_stage3_calibration_sweep\eval\dcas_ot_cal_p2_target.json`
- paired samples: `384`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| serendipity | 0.8566134367 | 0.8641977006 | +0.0075842639 | [0.004647, 0.010514] | 0.004975 | 0.249353 |
| cultural_calibration_kl | 1.5594957960 | 1.5501961971 | -0.0092995989 | [-0.009959, -0.008629] | 0.004975 | -1.360008 |
| minority_exposure_at_k | 0.7084635417 | 0.4997395833 | -0.2087239583 | [-0.217067, -0.199082] | 0.004975 | -2.374939 |

