# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `E:\Desktop\Echo\reports\benchmarks\v4_routeA_small_culturemert_lambdamart\eval\dcas_full_ot_calibrated_minor.json`
- candidate: `E:\Desktop\Echo\reports\benchmarks\v4_routeA_small_culturemert_lambdamart\eval\dcas_full_ot_calibrated_target.json`
- paired samples: `384`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| serendipity | 0.8378096758 | 0.8468967773 | +0.0090871014 | [0.005543, 0.012481] | 0.004975 | 0.286351 |
| cultural_calibration_kl | 1.1168723101 | 1.0872535167 | -0.0296187934 | [-0.034295, -0.024935] | 0.004975 | -0.687217 |
| minority_exposure_at_k | 0.6540364583 | 0.4427083333 | -0.2113281250 | [-0.216276, -0.205465] | 0.004975 | -3.802099 |

