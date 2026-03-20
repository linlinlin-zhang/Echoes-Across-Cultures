# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `E:\Desktop\Echo\reports\benchmarks\v4_routeA_small_culturemert_lambdamart\eval\bpr_listwise_hybrid.json`
- candidate: `E:\Desktop\Echo\reports\benchmarks\v4_routeA_small_culturemert_lambdamart\eval\dcas_full_ot_calibrated_target.json`
- paired samples: `384`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| serendipity | 0.5128268840 | 0.8468967773 | +0.3340698933 | [0.317253, 0.351750] | 0.004975 | 1.944907 |
| cultural_calibration_kl | 1.1112520705 | 1.0872535167 | -0.0239985538 | [-0.030045, -0.018085] | 0.004975 | -0.424815 |
| minority_exposure_at_k | 0.2003906250 | 0.4427083333 | +0.2423177083 | [0.227204, 0.258743] | 0.004975 | 1.689911 |

