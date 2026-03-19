# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `E:\Desktop\Echo\reports\benchmarks\public_routeA_phase2_cn_lambdamart\eval\knn.json`
- candidate: `E:\Desktop\Echo\reports\benchmarks\public_routeA_phase2_cn_lambdamart\eval\dcas_full_ot_calibrated_target.json`
- paired samples: `320`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| serendipity | 0.6683533523 | 0.7428681342 | +0.0745147819 | [0.060214, 0.090735] | 0.004975 | 0.524527 |
| cultural_calibration_kl | 1.1443106369 | 1.0498578383 | -0.0944527987 | [-0.113598, -0.075222] | 0.004975 | -0.484104 |
| minority_exposure_at_k | 0.2342187500 | 0.3740625000 | +0.1398437500 | [0.125773, 0.153305] | 0.004975 | 1.167158 |

