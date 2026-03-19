# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `E:\Desktop\Echo\reports\benchmarks\v3_main_culturemert_stage3_stronghybrid\eval\dcas_full_knn_open.json`
- candidate: `E:\Desktop\Echo\reports\benchmarks\v3_main_culturemert_stage3_stronghybrid\eval\dcas_full_ot.json`
- paired samples: `2400`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| serendipity | 0.2632374546 | 0.8448537518 | +0.5816162972 | [0.579071, 0.584089] | 0.003322 | 9.813306 |
| cultural_calibration_kl | 2.2987603883 | 2.0432477311 | -0.2555126573 | [-0.268502, -0.242242] | 0.003322 | -0.846830 |
| minority_exposure_at_k | 0.3735833333 | 0.2392083333 | -0.1343750000 | [-0.139867, -0.128969] | 0.003322 | -0.876779 |

