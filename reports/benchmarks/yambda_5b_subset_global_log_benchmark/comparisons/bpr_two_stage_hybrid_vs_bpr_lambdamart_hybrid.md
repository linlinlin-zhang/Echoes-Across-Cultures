# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `E:\Desktop\Echo\reports\benchmarks\yambda_5b_subset_global_log_benchmark\eval\bpr_two_stage_hybrid.json`
- candidate: `E:\Desktop\Echo\reports\benchmarks\yambda_5b_subset_global_log_benchmark\eval\bpr_lambdamart_hybrid.json`
- paired samples: `58`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| recall_at_10 | 0.4482758621 | 0.4655172414 | +0.0172413793 | [-0.051724, 0.103448] | 1.000000 | 0.058314 |
| recall_at_20 | 0.5344827586 | 0.5517241379 | +0.0172413793 | [0.000000, 0.051724] | 1.000000 | 0.131306 |
| ndcg_at_10 | 0.2100456562 | 0.2256970568 | +0.0156514006 | [-0.013945, 0.053809] | 0.383085 | 0.110515 |
| ndcg_at_20 | 0.2323077804 | 0.2478170467 | +0.0155092663 | [-0.012376, 0.046328] | 0.348259 | 0.133094 |
| mrr_at_10 | 0.1388820471 | 0.1534209086 | +0.0145388615 | [-0.025867, 0.051587] | 0.457711 | 0.100164 |
| mrr_at_20 | 0.1452735483 | 0.1596533734 | +0.0143798252 | [-0.019939, 0.052594] | 0.452736 | 0.101208 |

