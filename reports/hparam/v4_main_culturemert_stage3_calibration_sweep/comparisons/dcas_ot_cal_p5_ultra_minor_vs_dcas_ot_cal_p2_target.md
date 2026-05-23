# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `E:\Desktop\Echo\reports\hparam\v4_main_culturemert_stage3_calibration_sweep\eval\dcas_ot_cal_p5_ultra_minor.json`
- candidate: `E:\Desktop\Echo\reports\hparam\v4_main_culturemert_stage3_calibration_sweep\eval\dcas_ot_cal_p2_target.json`
- paired samples: `2400`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| serendipity | 0.8296226631 | 0.8315641066 | +0.0019414435 | [0.000706, 0.003046] | 0.004975 | 0.067461 |
| cultural_calibration_kl | 2.0527046180 | 2.0296378083 | -0.0230668098 | [-0.023795, -0.022072] | 0.004975 | -0.919646 |
| minority_exposure_at_k | 0.5837916667 | 0.4023333333 | -0.1814583333 | [-0.184821, -0.178645] | 0.004975 | -2.262248 |

