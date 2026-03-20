# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `E:\Desktop\Echo\reports\benchmarks\v4_main_culturemert_stage3_lambdamart\eval\cosine.json`
- candidate: `E:\Desktop\Echo\reports\benchmarks\v4_main_culturemert_stage3_lambdamart\eval\dcas_full_ot_calibrated_target.json`
- paired samples: `2400`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| serendipity | 0.6333239234 | 0.8315641066 | +0.1982401832 | [0.189985, 0.205596] | 0.004975 | 1.011131 |
| cultural_calibration_kl | 2.2333765023 | 2.0296378083 | -0.2037386940 | [-0.214362, -0.193527] | 0.004975 | -0.712110 |
| minority_exposure_at_k | 0.2206666667 | 0.4023333333 | +0.1816666667 | [0.176540, 0.185671] | 0.004975 | 1.525688 |

