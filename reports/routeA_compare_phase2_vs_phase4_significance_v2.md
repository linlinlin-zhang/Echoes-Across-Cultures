# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `reports/routeA_phase2_cn_eval_v2.json`
- candidate: `reports/routeA_phase4_tc_hsic_eval_v2.json`
- paired samples: `320`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| serendipity | 0.8109801431 | 0.8726107828 | +0.0616306397 | [0.052786, 0.070391] | 0.000500 | 0.765003 |
| cultural_calibration_kl | 1.6959370107 | 1.6967106313 | +0.0007736206 | [0.000647, 0.000905] | 0.000500 | 0.670887 |
| target_culture_prob_mean | 0.2503342772 | 0.2501426871 | -0.0001915901 | [-0.000223, -0.000161] | 0.000500 | -0.671280 |
| user_culture_alignment_kl | 0.0000011179 | 0.0000001482 | -0.0000009697 | [-0.000001, -0.000001] | 0.000500 | -0.848481 |

