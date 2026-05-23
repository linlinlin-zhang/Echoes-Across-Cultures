# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `E:\Desktop\Echo\reports\benchmarks\v2_main_culturemert\eval\cosine.json`
- candidate: `E:\Desktop\Echo\reports\benchmarks\v2_main_culturemert\eval\dcas_full_ot.json`
- paired samples: `600`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| serendipity | 0.6914276882 | 0.8378055970 | +0.1463779088 | [0.131419, 0.161744] | 0.003322 | 0.722737 |
| cultural_calibration_kl | 1.1817617460 | 0.8052499554 | -0.3765117906 | [-0.409823, -0.348352] | 0.003322 | -0.974158 |
| minority_exposure_at_k | 0.3445000000 | 0.3400833333 | -0.0044166667 | [-0.009544, 0.002377] | 0.112957 | -0.060562 |

