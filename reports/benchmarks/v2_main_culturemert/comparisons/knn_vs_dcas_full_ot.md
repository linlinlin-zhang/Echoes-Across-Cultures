# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `E:\Desktop\Echo\reports\benchmarks\v2_main_culturemert\eval\knn.json`
- candidate: `E:\Desktop\Echo\reports\benchmarks\v2_main_culturemert\eval\dcas_full_ot.json`
- paired samples: `600`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| serendipity | 0.7242404754 | 0.8378055970 | +0.1135651217 | [0.099439, 0.127581] | 0.003322 | 0.598142 |
| cultural_calibration_kl | 1.1672259217 | 0.8052499554 | -0.3619759663 | [-0.394649, -0.335311] | 0.003322 | -0.974747 |
| minority_exposure_at_k | 0.3507500000 | 0.3400833333 | -0.0106666667 | [-0.016127, -0.004456] | 0.003322 | -0.143672 |

