# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `E:\Desktop\Echo\reports\benchmarks\v4_routeA_small_culturemert_lambdamart\eval\bpr_mf.json`
- candidate: `E:\Desktop\Echo\reports\benchmarks\v4_routeA_small_culturemert_lambdamart\eval\dcas_full_ot_calibrated_target.json`
- paired samples: `384`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| serendipity | 0.4863807858 | 0.8468967773 | +0.3605159915 | [0.343748, 0.377548] | 0.000999 | 2.149341 |
| cultural_calibration_kl | 1.1223345233 | 1.0872535167 | -0.0350810065 | [-0.040406, -0.029518] | 0.000999 | -0.631270 |
| minority_exposure_at_k | 0.1027343750 | 0.4427083333 | +0.3399739583 | [0.327083, 0.353128] | 0.000999 | 2.738265 |

