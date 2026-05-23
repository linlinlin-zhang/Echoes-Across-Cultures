# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `E:\Desktop\Echo\reports\benchmarks\public_routeA_phase2_cn_lambdamart\eval\dcas_full_ot.json`
- candidate: `E:\Desktop\Echo\reports\benchmarks\public_routeA_phase2_cn_lambdamart\eval\dcas_full_ot_calibrated_target.json`
- paired samples: `320`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| serendipity | 0.7653383203 | 0.7428681342 | -0.0224701861 | [-0.028760, -0.014657] | 0.004975 | -0.320393 |
| cultural_calibration_kl | 1.2383578342 | 1.0498578383 | -0.1884999960 | [-0.205054, -0.173682] | 0.004975 | -1.220764 |
| minority_exposure_at_k | 0.2660937500 | 0.3740625000 | +0.1079687500 | [0.100141, 0.115164] | 0.004975 | 1.364536 |

