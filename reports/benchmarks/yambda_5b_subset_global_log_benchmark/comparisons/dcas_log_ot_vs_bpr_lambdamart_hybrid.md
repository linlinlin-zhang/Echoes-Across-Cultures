# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `E:\Desktop\Echo\reports\benchmarks\yambda_5b_subset_global_log_benchmark\eval\dcas_log_ot.json`
- candidate: `E:\Desktop\Echo\reports\benchmarks\yambda_5b_subset_global_log_benchmark\eval\bpr_lambdamart_hybrid.json`
- paired samples: `58`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| recall_at_10 | 0.0344827586 | 0.4655172414 | +0.4310344828 | [0.310345, 0.551724] | 0.004975 | 0.862852 |
| recall_at_20 | 0.0862068966 | 0.5517241379 | +0.4655172414 | [0.327155, 0.620690] | 0.004975 | 0.818650 |
| ndcg_at_10 | 0.0229885057 | 0.2256970568 | +0.2027085510 | [0.147010, 0.273456] | 0.004975 | 0.776988 |
| ndcg_at_20 | 0.0367393765 | 0.2478170467 | +0.2110776702 | [0.147823, 0.267951] | 0.004975 | 0.810144 |
| mrr_at_10 | 0.0197044335 | 0.1534209086 | +0.1337164751 | [0.082398, 0.198429] | 0.004975 | 0.609230 |
| mrr_at_20 | 0.0238296187 | 0.1596533734 | +0.1358237548 | [0.090444, 0.188948] | 0.004975 | 0.627783 |

