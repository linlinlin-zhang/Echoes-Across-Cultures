# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `reports/routeA_phase2_cn_eval.json`
- candidate: `reports/routeA_phase4_tc_hsic_eval.json`
- paired samples: `320`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| serendipity | 0.8109801431 | 0.8726107828 | +0.0616306397 | [0.052786, 0.070391] | 0.000500 | 0.765003 |
| cultural_calibration_kl | 1.3862943611 | 1.3862943611 | +0.0000000000 | [0.000000, 0.000000] | 1.000000 | 0.000000 |

