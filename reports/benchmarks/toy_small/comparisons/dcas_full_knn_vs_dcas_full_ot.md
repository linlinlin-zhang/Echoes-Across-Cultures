# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `E:\Desktop\Echo\reports\benchmarks\toy_small\eval\dcas_full_knn.json`
- candidate: `E:\Desktop\Echo\reports\benchmarks\toy_small\eval\dcas_full_ot.json`
- paired samples: `18`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| serendipity | 0.7977586821 | 0.7978846578 | +0.0001259758 | [-0.034950, 0.027301] | 1.000000 | 0.001799 |
| cultural_calibration_kl | 0.0423395355 | 0.0426531376 | +0.0003136021 | [-0.000567, 0.001388] | 0.597015 | 0.129015 |
| minority_exposure_at_k | 0.1666666667 | 0.1277777778 | -0.0388888889 | [-0.088889, 0.016667] | 0.124378 | -0.356074 |

