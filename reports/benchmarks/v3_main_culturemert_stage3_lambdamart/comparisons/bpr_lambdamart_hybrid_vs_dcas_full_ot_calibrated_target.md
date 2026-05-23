# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `E:\Desktop\Echo\reports\benchmarks\v3_main_culturemert_stage3_lambdamart\eval\bpr_lambdamart_hybrid.json`
- candidate: `E:\Desktop\Echo\reports\benchmarks\v3_main_culturemert_stage3_lambdamart\eval\dcas_full_ot_calibrated_target.json`
- paired samples: `2400`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| serendipity | 0.5106844154 | 0.8385876200 | +0.3279032046 | [0.322650, 0.332518] | 0.004975 | 2.296047 |
| cultural_calibration_kl | 1.9967404077 | 1.8792946220 | -0.1174457857 | [-0.127707, -0.108680] | 0.004975 | -0.564680 |
| minority_exposure_at_k | 0.2662083333 | 0.3814375000 | +0.1152291667 | [0.106921, 0.123053] | 0.004975 | 0.603405 |

