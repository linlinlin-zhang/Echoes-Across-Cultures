# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `E:\Desktop\Echo\reports\benchmarks\public_routeA_phase2_cn_lambdamart\eval\bpr_mf.json`
- candidate: `E:\Desktop\Echo\reports\benchmarks\public_routeA_phase2_cn_lambdamart\eval\dcas_full_ot_calibrated_target.json`
- paired samples: `320`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| serendipity | 0.6115913421 | 0.7428681342 | +0.1312767921 | [0.117392, 0.144323] | 0.004975 | 1.076162 |
| cultural_calibration_kl | 1.0096088020 | 1.0498578383 | +0.0402490363 | [0.020103, 0.058808] | 0.004975 | 0.218724 |
| minority_exposure_at_k | 0.1364062500 | 0.3740625000 | +0.2376562500 | [0.211559, 0.262508] | 0.004975 | 1.074903 |

