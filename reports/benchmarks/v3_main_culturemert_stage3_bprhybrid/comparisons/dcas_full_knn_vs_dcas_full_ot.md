# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `E:\Desktop\Echo\reports\benchmarks\v3_main_culturemert_stage3_bprhybrid\eval\dcas_full_knn.json`
- candidate: `E:\Desktop\Echo\reports\benchmarks\v3_main_culturemert_stage3_bprhybrid\eval\dcas_full_ot.json`
- paired samples: `2400`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| serendipity | 0.8454766807 | 0.8452264257 | -0.0002502549 | [-0.000909, 0.000366] | 0.552239 | -0.015521 |
| cultural_calibration_kl | 2.0416618609 | 2.0429787993 | +0.0013169383 | [0.000468, 0.002024] | 0.004975 | 0.058910 |
| minority_exposure_at_k | 0.2395000000 | 0.2397916667 | +0.0002916667 | [-0.000730, 0.001229] | 0.611940 | 0.010893 |

