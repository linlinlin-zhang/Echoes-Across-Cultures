# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `E:\Desktop\Echo\reports\benchmarks\v2_main_culturemert\eval\dcas_full_knn.json`
- candidate: `E:\Desktop\Echo\reports\benchmarks\v2_main_culturemert\eval\dcas_full_ot.json`
- paired samples: `600`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| serendipity | 0.8379375280 | 0.8378055970 | -0.0001319310 | [-0.001828, 0.001519] | 0.853821 | -0.006698 |
| cultural_calibration_kl | 0.8048160663 | 0.8052499554 | +0.0004338891 | [-0.001153, 0.002239] | 0.644518 | 0.020356 |
| minority_exposure_at_k | 0.3406666667 | 0.3400833333 | -0.0005833333 | [-0.002167, 0.000917] | 0.451827 | -0.031742 |

