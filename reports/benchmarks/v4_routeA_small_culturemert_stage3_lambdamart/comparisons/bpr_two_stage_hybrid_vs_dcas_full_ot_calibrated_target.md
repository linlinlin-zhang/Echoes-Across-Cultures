# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `E:\Desktop\Echo\reports\benchmarks\v4_routeA_small_culturemert_stage3_lambdamart\eval\bpr_two_stage_hybrid.json`
- candidate: `E:\Desktop\Echo\reports\benchmarks\v4_routeA_small_culturemert_stage3_lambdamart\eval\dcas_full_ot_calibrated_target.json`
- paired samples: `384`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| serendipity | 0.5002452011 | 0.8406298893 | +0.3403846882 | [0.324635, 0.356934] | 0.000999 | 2.101821 |
| cultural_calibration_kl | 1.1264192366 | 1.0212990236 | -0.1051202130 | [-0.113185, -0.096760] | 0.000999 | -1.323770 |
| minority_exposure_at_k | 0.1787760417 | 0.5095052083 | +0.3307291667 | [0.318617, 0.344404] | 0.000999 | 2.499165 |

