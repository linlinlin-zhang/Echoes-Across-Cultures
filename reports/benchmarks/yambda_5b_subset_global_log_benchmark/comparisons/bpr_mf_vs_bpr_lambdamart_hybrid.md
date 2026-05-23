# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `E:\Desktop\Echo\reports\benchmarks\yambda_5b_subset_global_log_benchmark\eval\bpr_mf.json`
- candidate: `E:\Desktop\Echo\reports\benchmarks\yambda_5b_subset_global_log_benchmark\eval\bpr_lambdamart_hybrid.json`
- paired samples: `58`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| recall_at_10 | 0.3965517241 | 0.4655172414 | +0.0689655172 | [0.000000, 0.155172] | 0.223881 | 0.217628 |
| recall_at_20 | 0.5172413793 | 0.5517241379 | +0.0344827586 | [0.000000, 0.086207] | 0.557214 | 0.187346 |
| ndcg_at_10 | 0.1861089247 | 0.2256970568 | +0.0395881321 | [0.007727, 0.076796] | 0.009950 | 0.307386 |
| ndcg_at_20 | 0.2177773721 | 0.2478170467 | +0.0300396746 | [0.013005, 0.049265] | 0.004975 | 0.412118 |
| mrr_at_10 | 0.1245621237 | 0.1534209086 | +0.0288587849 | [0.009033, 0.051799] | 0.004975 | 0.339061 |
| mrr_at_20 | 0.1338579116 | 0.1596533734 | +0.0257954619 | [0.010559, 0.040538] | 0.004975 | 0.378173 |

