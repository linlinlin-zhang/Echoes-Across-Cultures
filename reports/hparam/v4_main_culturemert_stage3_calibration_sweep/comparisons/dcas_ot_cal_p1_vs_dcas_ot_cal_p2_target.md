# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `E:\Desktop\Echo\reports\hparam\v4_main_culturemert_stage3_calibration_sweep\eval\dcas_ot_cal_p1.json`
- candidate: `E:\Desktop\Echo\reports\hparam\v4_main_culturemert_stage3_calibration_sweep\eval\dcas_ot_cal_p2_target.json`
- paired samples: `2400`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| serendipity | 0.8370891096 | 0.8315641066 | -0.0055250030 | [-0.006568, -0.004652] | 0.004975 | -0.202871 |
| cultural_calibration_kl | 2.0228399940 | 2.0296378083 | +0.0067978143 | [0.006188, 0.007498] | 0.004975 | 0.455132 |
| minority_exposure_at_k | 0.3479166667 | 0.4023333333 | +0.0544166667 | [0.052681, 0.056167] | 0.004975 | 1.111202 |

