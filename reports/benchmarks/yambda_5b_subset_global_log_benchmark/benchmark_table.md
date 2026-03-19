# Log Benchmark: yambda_5b_subset_global_log_benchmark

- tracks: `E:\Desktop\Echo\storage\public\yambda_5b_subset\tracks.npz`
- interactions: `E:\Desktop\Echo\storage\public\yambda_5b_subset\interactions.csv`
- eval_users: `58`
- eval_cases: `58`

| method | Recall@10 | Recall@20 | NDCG@10 | NDCG@20 | MRR@10 | MRR@20 |
|---|---:|---:|---:|---:|---:|---:|
| popularity | 0.0862 | 0.1552 | 0.0585 | 0.0770 | 0.0499 | 0.0555 |
| cosine | 0.1379 | 0.2069 | 0.0773 | 0.0939 | 0.0579 | 0.0621 |
| knn | 0.1897 | 0.2931 | 0.1148 | 0.1397 | 0.0904 | 0.0966 |
| bpr_mf | 0.3966 | 0.5172 | 0.1861 | 0.2178 | 0.1246 | 0.1339 |
| bpr_two_stage_hybrid | 0.4483 | 0.5345 | 0.2100 | 0.2323 | 0.1389 | 0.1453 |
| bpr_lambdamart_hybrid | 0.4655 | 0.5517 | 0.2257 | 0.2478 | 0.1534 | 0.1597 |
| dcas_log_ot | 0.0345 | 0.0862 | 0.0230 | 0.0367 | 0.0197 | 0.0238 |

## Comparisons vs Reference

| method | metric | delta_mean(reference - base) | p_value |
|---|---|---:|---:|
| bpr_mf | recall_at_10 | +0.068966 | 0.223881 |
| bpr_mf | recall_at_20 | +0.034483 | 0.557214 |
| bpr_mf | ndcg_at_10 | +0.039588 | 0.009950 |
| bpr_mf | ndcg_at_20 | +0.030040 | 0.004975 |
| bpr_mf | mrr_at_10 | +0.028859 | 0.004975 |
| bpr_mf | mrr_at_20 | +0.025795 | 0.004975 |
| bpr_two_stage_hybrid | recall_at_10 | +0.017241 | 1.000000 |
| bpr_two_stage_hybrid | recall_at_20 | +0.017241 | 1.000000 |
| bpr_two_stage_hybrid | ndcg_at_10 | +0.015651 | 0.383085 |
| bpr_two_stage_hybrid | ndcg_at_20 | +0.015509 | 0.348259 |
| bpr_two_stage_hybrid | mrr_at_10 | +0.014539 | 0.457711 |
| bpr_two_stage_hybrid | mrr_at_20 | +0.014380 | 0.452736 |
| cosine | recall_at_10 | +0.327586 | 0.004975 |
| cosine | recall_at_20 | +0.344828 | 0.004975 |
| cosine | ndcg_at_10 | +0.148422 | 0.004975 |
| cosine | ndcg_at_20 | +0.153915 | 0.004975 |
| cosine | mrr_at_10 | +0.095519 | 0.004975 |
| cosine | mrr_at_20 | +0.097592 | 0.009950 |
| dcas_log_ot | recall_at_10 | +0.431034 | 0.004975 |
| dcas_log_ot | recall_at_20 | +0.465517 | 0.004975 |
| dcas_log_ot | ndcg_at_10 | +0.202709 | 0.004975 |
| dcas_log_ot | ndcg_at_20 | +0.211078 | 0.004975 |
| dcas_log_ot | mrr_at_10 | +0.133716 | 0.004975 |
| dcas_log_ot | mrr_at_20 | +0.135824 | 0.004975 |
| knn | recall_at_10 | +0.275862 | 0.004975 |
| knn | recall_at_20 | +0.258621 | 0.004975 |
| knn | ndcg_at_10 | +0.110926 | 0.009950 |
| knn | ndcg_at_20 | +0.108081 | 0.009950 |
| knn | mrr_at_10 | +0.063047 | 0.004975 |
| knn | mrr_at_20 | +0.063062 | 0.019900 |
| popularity | recall_at_10 | +0.379310 | 0.004975 |
| popularity | recall_at_20 | +0.396552 | 0.004975 |
| popularity | ndcg_at_10 | +0.167164 | 0.004975 |
| popularity | ndcg_at_20 | +0.170866 | 0.004975 |
| popularity | mrr_at_10 | +0.103544 | 0.009950 |
| popularity | mrr_at_20 | +0.104167 | 0.004975 |
