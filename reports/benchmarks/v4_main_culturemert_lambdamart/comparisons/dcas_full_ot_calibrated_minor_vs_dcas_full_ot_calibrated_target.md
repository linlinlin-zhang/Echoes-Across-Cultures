# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `E:\Desktop\Echo\reports\benchmarks\v4_main_culturemert_lambdamart\eval\dcas_full_ot_calibrated_minor.json`
- candidate: `E:\Desktop\Echo\reports\benchmarks\v4_main_culturemert_lambdamart\eval\dcas_full_ot_calibrated_target.json`
- paired samples: `2400`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| serendipity | 0.8075922131 | 0.8110557968 | +0.0034635836 | [0.002606, 0.004398] | 0.000999 | 0.157312 |
| cultural_calibration_kl | 2.0618796244 | 2.0492074270 | -0.0126721973 | [-0.013293, -0.011988] | 0.000999 | -0.778411 |
| minority_exposure_at_k | 0.5060625000 | 0.3998125000 | -0.1062500000 | [-0.108937, -0.103583] | 0.000999 | -1.594945 |

