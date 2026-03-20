# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `E:\Desktop\Echo\reports\benchmarks\v4_main_culturemert_stage3_lambdamart\eval\lightfm_like.json`
- candidate: `E:\Desktop\Echo\reports\benchmarks\v4_main_culturemert_stage3_lambdamart\eval\dcas_full_ot_calibrated_target.json`
- paired samples: `2400`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| serendipity | 0.5026620134 | 0.8315641066 | +0.3289020932 | [0.319985, 0.336287] | 0.004975 | 1.850209 |
| cultural_calibration_kl | 2.1850225946 | 2.0296378083 | -0.1553847863 | [-0.162875, -0.147735] | 0.004975 | -0.684907 |
| minority_exposure_at_k | 0.1336250000 | 0.4023333333 | +0.2687083333 | [0.262913, 0.273772] | 0.004975 | 1.817720 |

