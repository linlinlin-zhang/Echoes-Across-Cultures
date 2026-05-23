# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `E:\Desktop\Echo\reports\benchmarks\public_routeA_phase2_cn_lambdamart\eval\bpr_two_stage_hybrid.json`
- candidate: `E:\Desktop\Echo\reports\benchmarks\public_routeA_phase2_cn_lambdamart\eval\dcas_full_ot_calibrated_target.json`
- paired samples: `320`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| serendipity | 0.6222604693 | 0.7428681342 | +0.1206076650 | [0.108765, 0.133245] | 0.004975 | 1.182626 |
| cultural_calibration_kl | 1.0711666486 | 1.0498578383 | -0.0213088103 | [-0.034340, -0.010131] | 0.014925 | -0.168161 |
| minority_exposure_at_k | 0.2093750000 | 0.3740625000 | +0.1646875000 | [0.144969, 0.179930] | 0.004975 | 1.066895 |

