# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `E:\Desktop\Echo\reports\benchmarks\yambda_5b_subset_global_log_benchmark\eval\popularity.json`
- candidate: `E:\Desktop\Echo\reports\benchmarks\yambda_5b_subset_global_log_benchmark\eval\bpr_lambdamart_hybrid.json`
- paired samples: `58`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| recall_at_10 | 0.0862068966 | 0.4655172414 | +0.3793103448 | [0.258621, 0.517241] | 0.004975 | 0.723774 |
| recall_at_20 | 0.1551724138 | 0.5517241379 | +0.3965517241 | [0.258190, 0.534483] | 0.004975 | 0.751314 |
| ndcg_at_10 | 0.0585334422 | 0.2256970568 | +0.1671636146 | [0.090357, 0.238641] | 0.004975 | 0.563305 |
| ndcg_at_20 | 0.0769508051 | 0.2478170467 | +0.1708662417 | [0.096535, 0.241339] | 0.004975 | 0.603802 |
| mrr_at_10 | 0.0498768473 | 0.1534209086 | +0.1035440613 | [0.044519, 0.175974] | 0.009950 | 0.401288 |
| mrr_at_20 | 0.0554864827 | 0.1596533734 | +0.1041668908 | [0.055161, 0.164238] | 0.004975 | 0.410861 |

