# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `E:\Desktop\Echo\reports\benchmarks\v4_routeA_small_culturemert_stage3_lambdamart\eval\cosine.json`
- candidate: `E:\Desktop\Echo\reports\benchmarks\v4_routeA_small_culturemert_stage3_lambdamart\eval\dcas_full_ot_calibrated_target.json`
- paired samples: `384`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| serendipity | 0.4303718807 | 0.8406298893 | +0.4102580086 | [0.384486, 0.430628] | 0.004975 | 1.905062 |
| cultural_calibration_kl | 1.1607639461 | 1.0212990236 | -0.1394649225 | [-0.153095, -0.130190] | 0.004975 | -1.294387 |
| minority_exposure_at_k | 0.2121093750 | 0.5095052083 | +0.2973958333 | [0.286706, 0.307695] | 0.004975 | 3.155309 |

