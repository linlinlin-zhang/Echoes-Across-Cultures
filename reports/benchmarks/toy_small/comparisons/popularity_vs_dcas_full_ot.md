# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `E:\Desktop\Echo\reports\benchmarks\toy_small\eval\popularity.json`
- candidate: `E:\Desktop\Echo\reports\benchmarks\toy_small\eval\dcas_full_ot.json`
- paired samples: `18`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| serendipity | 0.6407683152 | 0.7978846578 | +0.1571163427 | [0.111514, 0.194827] | 0.004975 | 1.681881 |
| cultural_calibration_kl | 0.0425794531 | 0.0426531376 | +0.0000736846 | [-0.001531, 0.001711] | 0.935323 | 0.021082 |
| minority_exposure_at_k | 0.0000000000 | 0.1277777778 | +0.1277777778 | [0.061111, 0.222222] | 0.004975 | 0.747981 |

