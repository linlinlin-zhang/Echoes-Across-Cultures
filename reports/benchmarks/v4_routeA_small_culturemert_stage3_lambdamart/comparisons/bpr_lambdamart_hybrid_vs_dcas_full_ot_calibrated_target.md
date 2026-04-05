# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `E:\Desktop\Echo\reports\benchmarks\v4_routeA_small_culturemert_stage3_lambdamart\eval\bpr_lambdamart_hybrid.json`
- candidate: `E:\Desktop\Echo\reports\benchmarks\v4_routeA_small_culturemert_stage3_lambdamart\eval\dcas_full_ot_calibrated_target.json`
- paired samples: `384`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| serendipity | 0.5017876630 | 0.8406298893 | +0.3388422263 | [0.322136, 0.355966] | 0.000999 | 2.049099 |
| cultural_calibration_kl | 1.1136285667 | 1.0212990236 | -0.0923295431 | [-0.099369, -0.085442] | 0.000999 | -1.274002 |
| minority_exposure_at_k | 0.1572916667 | 0.5095052083 | +0.3522135417 | [0.339967, 0.365625] | 0.000999 | 2.729885 |

