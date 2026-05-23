# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `E:\Desktop\Echo\reports\benchmarks\yambda_5b_subset_global_log_benchmark\eval\knn.json`
- candidate: `E:\Desktop\Echo\reports\benchmarks\yambda_5b_subset_global_log_benchmark\eval\bpr_lambdamart_hybrid.json`
- paired samples: `58`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| recall_at_10 | 0.1896551724 | 0.4655172414 | +0.2758620690 | [0.154741, 0.413793] | 0.004975 | 0.527545 |
| recall_at_20 | 0.2931034483 | 0.5517241379 | +0.2586206897 | [0.120690, 0.396983] | 0.004975 | 0.471889 |
| ndcg_at_10 | 0.1147709182 | 0.2256970568 | +0.1109261386 | [0.055215, 0.165869] | 0.009950 | 0.453098 |
| ndcg_at_20 | 0.1397358584 | 0.2478170467 | +0.1080811883 | [0.043297, 0.164261] | 0.009950 | 0.455311 |
| mrr_at_10 | 0.0903735632 | 0.1534209086 | +0.0630473454 | [0.016837, 0.109667] | 0.004975 | 0.335311 |
| mrr_at_20 | 0.0965918642 | 0.1596533734 | +0.0630615092 | [0.019755, 0.103625] | 0.019900 | 0.341826 |

