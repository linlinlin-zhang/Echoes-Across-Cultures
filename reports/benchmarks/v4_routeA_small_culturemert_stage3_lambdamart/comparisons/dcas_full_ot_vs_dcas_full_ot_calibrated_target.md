# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `E:\Desktop\Echo\reports\benchmarks\v4_routeA_small_culturemert_stage3_lambdamart\eval\dcas_full_ot.json`
- candidate: `E:\Desktop\Echo\reports\benchmarks\v4_routeA_small_culturemert_stage3_lambdamart\eval\dcas_full_ot_calibrated_target.json`
- paired samples: `384`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| serendipity | 0.8501348013 | 0.8406298893 | -0.0095049120 | [-0.014261, -0.005062] | 0.004975 | -0.208168 |
| cultural_calibration_kl | 1.1439147723 | 1.0212990236 | -0.1226157487 | [-0.135682, -0.109313] | 0.004975 | -1.085456 |
| minority_exposure_at_k | 0.3027343750 | 0.5095052083 | +0.2067708333 | [0.199089, 0.215762] | 0.004975 | 2.445794 |

