# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `E:\Desktop\Echo\reports\benchmarks\public_routeA_phase2_cn_lambdamart\eval\cosine.json`
- candidate: `E:\Desktop\Echo\reports\benchmarks\public_routeA_phase2_cn_lambdamart\eval\dcas_full_ot_calibrated_target.json`
- paired samples: `320`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| serendipity | 0.6338521312 | 0.7428681342 | +0.1090160030 | [0.094015, 0.124756] | 0.004975 | 0.775920 |
| cultural_calibration_kl | 1.0880140954 | 1.0498578383 | -0.0381562572 | [-0.053591, -0.024531] | 0.004975 | -0.251995 |
| minority_exposure_at_k | 0.2504687500 | 0.3740625000 | +0.1235937500 | [0.111090, 0.137691] | 0.004975 | 1.027628 |

