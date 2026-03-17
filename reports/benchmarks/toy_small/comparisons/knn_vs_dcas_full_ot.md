# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `E:\Desktop\Echo\reports\benchmarks\toy_small\eval\knn.json`
- candidate: `E:\Desktop\Echo\reports\benchmarks\toy_small\eval\dcas_full_ot.json`
- paired samples: `18`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| serendipity | 0.7398026788 | 0.7978846578 | +0.0580819790 | [-0.020257, 0.116542] | 0.104478 | 0.391260 |
| cultural_calibration_kl | 0.0430266728 | 0.0426531376 | -0.0003735352 | [-0.002734, 0.001591] | 0.800995 | -0.076520 |
| minority_exposure_at_k | 0.2111111111 | 0.1277777778 | -0.0833333333 | [-0.155556, -0.016528] | 0.069652 | -0.527046 |

