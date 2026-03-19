# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `E:\Desktop\Echo\reports\benchmarks\public_routeA_phase2_cn_lambdamart\eval\popularity.json`
- candidate: `E:\Desktop\Echo\reports\benchmarks\public_routeA_phase2_cn_lambdamart\eval\dcas_full_ot_calibrated_target.json`
- paired samples: `320`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| serendipity | 0.5668424545 | 0.7428681342 | +0.1760256798 | [0.159186, 0.193726] | 0.004975 | 1.038776 |
| cultural_calibration_kl | 0.9961225998 | 1.0498578383 | +0.0537352385 | [0.032264, 0.075130] | 0.004975 | 0.278587 |
| minority_exposure_at_k | 0.0000000000 | 0.3740625000 | +0.3740625000 | [0.336094, 0.415023] | 0.004975 | 1.023554 |

