# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `E:\Desktop\Echo\reports\benchmarks\toy_small\eval\shallow_mlp.json`
- candidate: `E:\Desktop\Echo\reports\benchmarks\toy_small\eval\dcas_full_ot.json`
- paired samples: `18`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| serendipity | 0.6701452127 | 0.7978846578 | +0.1277394451 | [0.082678, 0.175176] | 0.004975 | 1.225803 |
| cultural_calibration_kl | 0.0430426259 | 0.0426531376 | -0.0003894883 | [-0.002118, 0.001510] | 0.736318 | -0.086537 |
| minority_exposure_at_k | 0.2111111111 | 0.1277777778 | -0.0833333333 | [-0.150000, -0.021944] | 0.064677 | -0.503868 |

