# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `E:\Desktop\Echo\reports\benchmarks\yambda_5b_subset_global_log_benchmark\eval\cosine.json`
- candidate: `E:\Desktop\Echo\reports\benchmarks\yambda_5b_subset_global_log_benchmark\eval\bpr_lambdamart_hybrid.json`
- paired samples: `58`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| recall_at_10 | 0.1379310345 | 0.4655172414 | +0.3275862069 | [0.206466, 0.465948] | 0.004975 | 0.603838 |
| recall_at_20 | 0.2068965517 | 0.5517241379 | +0.3448275862 | [0.189655, 0.500000] | 0.004975 | 0.595640 |
| ndcg_at_10 | 0.0772751681 | 0.2256970568 | +0.1484218886 | [0.090376, 0.223200] | 0.004975 | 0.519793 |
| ndcg_at_20 | 0.0939022853 | 0.2478170467 | +0.1539147615 | [0.092556, 0.224615] | 0.004975 | 0.559057 |
| mrr_at_10 | 0.0579022989 | 0.1534209086 | +0.0955186097 | [0.043961, 0.157334] | 0.004975 | 0.400904 |
| mrr_at_20 | 0.0620614035 | 0.1596533734 | +0.0975919699 | [0.050561, 0.161561] | 0.009950 | 0.417790 |

