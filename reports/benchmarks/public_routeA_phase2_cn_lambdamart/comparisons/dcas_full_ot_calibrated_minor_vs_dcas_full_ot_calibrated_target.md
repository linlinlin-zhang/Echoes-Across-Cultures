# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `E:\Desktop\Echo\reports\benchmarks\public_routeA_phase2_cn_lambdamart\eval\dcas_full_ot_calibrated_minor.json`
- candidate: `E:\Desktop\Echo\reports\benchmarks\public_routeA_phase2_cn_lambdamart\eval\dcas_full_ot_calibrated_target.json`
- paired samples: `320`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| serendipity | 0.7372331251 | 0.7428681342 | +0.0056350091 | [0.000260, 0.011219] | 0.054726 | 0.116237 |
| cultural_calibration_kl | 1.1020353819 | 1.0498578383 | -0.0521775436 | [-0.056987, -0.047939] | 0.004975 | -1.288089 |
| minority_exposure_at_k | 0.4634375000 | 0.3740625000 | -0.0893750000 | [-0.097508, -0.079527] | 0.004975 | -0.956724 |

