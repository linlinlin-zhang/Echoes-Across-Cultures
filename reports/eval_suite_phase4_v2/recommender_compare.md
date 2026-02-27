# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `reports\eval_suite_phase4_v2\recommender_eval_baseline.json`
- candidate: `reports\eval_suite_phase4_v2\recommender_eval.json`
- paired samples: `320`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| serendipity | 0.8109801431 | 0.8726107828 | +0.0616306397 | [0.052776, 0.070705] | 0.000500 | 0.765003 |
| cultural_calibration_kl | 1.6959370107 | 1.6967106313 | +0.0007736206 | [0.000654, 0.000903] | 0.000500 | 0.670887 |
| target_culture_prob_mean | 0.2503342772 | 0.2501426871 | -0.0001915901 | [-0.000222, -0.000162] | 0.000500 | -0.671280 |

