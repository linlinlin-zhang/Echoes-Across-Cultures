# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `E:\Desktop\Echo\reports\benchmarks\public_routeA_phase2_cn_lambdamart\eval\bpr_lambdamart_hybrid.json`
- candidate: `E:\Desktop\Echo\reports\benchmarks\public_routeA_phase2_cn_lambdamart\eval\dcas_full_ot_calibrated_target.json`
- paired samples: `320`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| serendipity | 0.6299612465 | 0.7428681342 | +0.1129068877 | [0.101146, 0.122668] | 0.004975 | 1.117548 |
| cultural_calibration_kl | 1.0016968874 | 1.0498578383 | +0.0481609509 | [0.029753, 0.061565] | 0.004975 | 0.301613 |
| minority_exposure_at_k | 0.2546875000 | 0.3740625000 | +0.1193750000 | [0.100918, 0.132688] | 0.004975 | 0.846442 |

