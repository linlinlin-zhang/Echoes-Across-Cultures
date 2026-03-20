# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `E:\Desktop\Echo\reports\benchmarks\v4_main_culturemert_stage3_lambdamart\eval\bpr_mf.json`
- candidate: `E:\Desktop\Echo\reports\benchmarks\v4_main_culturemert_stage3_lambdamart\eval\dcas_full_ot_calibrated_target.json`
- paired samples: `2400`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| serendipity | 0.5372903401 | 0.8315641066 | +0.2942737665 | [0.286787, 0.299811] | 0.004975 | 1.945871 |
| cultural_calibration_kl | 2.1146777049 | 2.0296378083 | -0.0850398966 | [-0.090466, -0.077573] | 0.004975 | -0.523316 |
| minority_exposure_at_k | 0.1646250000 | 0.4023333333 | +0.2377083333 | [0.231769, 0.244793] | 0.004975 | 1.445371 |

