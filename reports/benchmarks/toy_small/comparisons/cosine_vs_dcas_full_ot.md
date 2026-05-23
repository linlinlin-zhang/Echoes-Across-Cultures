# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `E:\Desktop\Echo\reports\benchmarks\toy_small\eval\cosine.json`
- candidate: `E:\Desktop\Echo\reports\benchmarks\toy_small\eval\dcas_full_ot.json`
- paired samples: `18`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| serendipity | 0.7080922279 | 0.7978846578 | +0.0897924299 | [0.018818, 0.150727] | 0.034826 | 0.582872 |
| cultural_calibration_kl | 0.0427679031 | 0.0426531376 | -0.0001147655 | [-0.002123, 0.001844] | 0.950249 | -0.026023 |
| minority_exposure_at_k | 0.2222222222 | 0.1277777778 | -0.0944444444 | [-0.144583, -0.033333] | 0.024876 | -0.723799 |

