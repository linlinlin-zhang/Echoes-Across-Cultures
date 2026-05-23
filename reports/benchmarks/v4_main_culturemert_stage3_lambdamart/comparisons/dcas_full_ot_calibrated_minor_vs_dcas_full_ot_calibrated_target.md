# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `E:\Desktop\Echo\reports\benchmarks\v4_main_culturemert_stage3_lambdamart\eval\dcas_full_ot_calibrated_minor.json`
- candidate: `E:\Desktop\Echo\reports\benchmarks\v4_main_culturemert_stage3_lambdamart\eval\dcas_full_ot_calibrated_target.json`
- paired samples: `2400`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| serendipity | 0.8281576285 | 0.8315641066 | +0.0034064782 | [0.002315, 0.004500] | 0.000999 | 0.126224 |
| cultural_calibration_kl | 2.0477411035 | 2.0296378083 | -0.0181032952 | [-0.018990, -0.017247] | 0.000999 | -0.809793 |
| minority_exposure_at_k | 0.5302708333 | 0.4023333333 | -0.1279375000 | [-0.130751, -0.125061] | 0.000999 | -1.859658 |

