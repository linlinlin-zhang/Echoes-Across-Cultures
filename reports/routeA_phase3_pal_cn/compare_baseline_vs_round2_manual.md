# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `reports/routeA_phase3_pal_cn/eval_round0_baseline.json`
- candidate: `reports/routeA_phase3_pal_cn/eval_round2.json`
- paired samples: `320`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| serendipity | 0.8109801431 | 0.7985860643 | -0.0123940788 | [-0.023392, -0.001334] | 0.031484 | -0.121828 |
| cultural_calibration_kl | 1.3862943611 | 1.3862943611 | +0.0000000000 | [0.000000, 0.000000] | 1.000000 | 0.000000 |

