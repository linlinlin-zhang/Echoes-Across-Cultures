# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `E:\Desktop\Echo\reports\benchmarks\public_routeA_phase2_cn_lambdamart\eval\bpr_listwise_hybrid.json`
- candidate: `E:\Desktop\Echo\reports\benchmarks\public_routeA_phase2_cn_lambdamart\eval\dcas_full_ot_calibrated_target.json`
- paired samples: `320`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| serendipity | 0.6371169574 | 0.7428681342 | +0.1057511768 | [0.094593, 0.116036] | 0.004975 | 1.080019 |
| cultural_calibration_kl | 1.0653539887 | 1.0498578383 | -0.0154961504 | [-0.029355, -0.001170] | 0.069652 | -0.104245 |
| minority_exposure_at_k | 0.2185937500 | 0.3740625000 | +0.1554687500 | [0.139055, 0.171098] | 0.004975 | 1.124468 |

