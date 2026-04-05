# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `E:\Desktop\Echo\reports\hparam\v4_routeA_small_gemini_stage3_calibration_sweep\eval\dcas_ot_cal_p1.json`
- candidate: `E:\Desktop\Echo\reports\hparam\v4_routeA_small_gemini_stage3_calibration_sweep\eval\dcas_ot_cal_p2_target.json`
- paired samples: `384`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| serendipity | 0.8667287657 | 0.8641977006 | -0.0025310651 | [-0.004441, -0.000548] | 0.039801 | -0.111819 |
| cultural_calibration_kl | 1.5486355631 | 1.5501961971 | +0.0015606340 | [0.001322, 0.001807] | 0.004975 | 0.584653 |
| minority_exposure_at_k | 0.4359375000 | 0.4997395833 | +0.0638020833 | [0.059115, 0.068493] | 0.004975 | 1.329649 |

