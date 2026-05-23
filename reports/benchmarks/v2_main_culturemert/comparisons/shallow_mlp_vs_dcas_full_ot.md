# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `E:\Desktop\Echo\reports\benchmarks\v2_main_culturemert\eval\shallow_mlp.json`
- candidate: `E:\Desktop\Echo\reports\benchmarks\v2_main_culturemert\eval\dcas_full_ot.json`
- paired samples: `600`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| serendipity | 0.6244889459 | 0.8378055970 | +0.2133166512 | [0.199464, 0.229423] | 0.003322 | 1.070461 |
| cultural_calibration_kl | 1.3420654626 | 0.8052499554 | -0.5368155072 | [-0.572297, -0.505541] | 0.003322 | -1.341308 |
| minority_exposure_at_k | 0.3440000000 | 0.3400833333 | -0.0039166667 | [-0.009508, 0.002500] | 0.182724 | -0.050889 |

