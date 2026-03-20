# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `E:\Desktop\Echo\reports\benchmarks\v4_routeA_small_culturemert_stage3_lambdamart\eval\bpr_two_stage_hybrid.json`
- candidate: `E:\Desktop\Echo\reports\benchmarks\v4_routeA_small_culturemert_stage3_lambdamart\eval\dcas_full_ot_calibrated_target.json`
- paired samples: `384`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| serendipity | 0.5002452011 | 0.8406298893 | +0.3403846882 | [0.325004, 0.355479] | 0.004975 | 2.101821 |
| cultural_calibration_kl | 1.1264192366 | 1.0212990236 | -0.1051202130 | [-0.113249, -0.097213] | 0.004975 | -1.323770 |
| minority_exposure_at_k | 0.1787760417 | 0.5095052083 | +0.3307291667 | [0.316787, 0.343750] | 0.004975 | 2.499165 |

