# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `E:\Desktop\Echo\reports\benchmarks\v3_main_culturemert_stage3_bprhybrid\eval\knn.json`
- candidate: `E:\Desktop\Echo\reports\benchmarks\v3_main_culturemert_stage3_bprhybrid\eval\dcas_full_ot.json`
- paired samples: `2400`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| serendipity | 0.5704624771 | 0.8452264257 | +0.2747639486 | [0.266569, 0.282134] | 0.004975 | 1.407014 |
| cultural_calibration_kl | 2.1846338018 | 2.0429787993 | -0.1416550025 | [-0.148481, -0.134287] | 0.004975 | -0.740301 |
| minority_exposure_at_k | 0.2263541667 | 0.2397916667 | +0.0134375000 | [0.009580, 0.016771] | 0.004975 | 0.139127 |

