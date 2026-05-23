# Recommender Benchmark: v3_main_culturemert_stage3_bprhybrid

- tracks: `E:/Desktop/Echo/storage/public/research_dataset_v3/tracks_culturemert_v3_main_mw3.npz`
- interactions: `E:\Desktop\Echo\storage\public\research_dataset_v3\interactions_v3_main_mixed_mw3.csv`
- reference_method: `dcas_full_ot`

| method | serendipity | calibration_kl | minority@k | target_prob |
|---|---:|---:|---:|---:|
| popularity | 0.454303 | 2.070947 | 0.000000 | 0.184229 |
| cosine | 0.554239 | 2.176031 | 0.226542 | 0.164475 |
| knn | 0.570462 | 2.184634 | 0.226354 | 0.162384 |
| hybrid_content_popularity_diversity | 0.536868 | 2.163824 | 0.070021 | 0.166311 |
| bpr_mf | 0.491591 | 2.022577 | 0.149146 | 0.195735 |
| bpr_two_stage_hybrid | 0.510193 | 2.008183 | 0.283750 | 0.200460 |
| dcas_full_ot | 0.845226 | 2.042979 | 0.239792 | 0.192066 |
| dcas_full_knn | 0.845477 | 2.041662 | 0.239500 | 0.192517 |

## Reference Comparisons

| baseline | metric | delta(reference-baseline) | p_value |
|---|---|---:|---:|
| popularity | serendipity | +0.390924 | 0.004975 |
| popularity | cultural_calibration_kl | -0.027969 | 0.004975 |
| popularity | minority_exposure_at_k | +0.239792 | 0.004975 |
| cosine | serendipity | +0.290987 | 0.004975 |
| cosine | cultural_calibration_kl | -0.133052 | 0.004975 |
| cosine | minority_exposure_at_k | +0.013250 | 0.004975 |
| knn | serendipity | +0.274764 | 0.004975 |
| knn | cultural_calibration_kl | -0.141655 | 0.004975 |
| knn | minority_exposure_at_k | +0.013437 | 0.004975 |
| hybrid_content_popularity_diversity | serendipity | +0.308358 | 0.004975 |
| hybrid_content_popularity_diversity | cultural_calibration_kl | -0.120846 | 0.004975 |
| hybrid_content_popularity_diversity | minority_exposure_at_k | +0.169771 | 0.004975 |
| bpr_mf | serendipity | +0.353635 | 0.004975 |
| bpr_mf | cultural_calibration_kl | +0.020402 | 0.004975 |
| bpr_mf | minority_exposure_at_k | +0.090646 | 0.004975 |
| bpr_two_stage_hybrid | serendipity | +0.335034 | 0.004975 |
| bpr_two_stage_hybrid | cultural_calibration_kl | +0.034796 | 0.004975 |
| bpr_two_stage_hybrid | minority_exposure_at_k | -0.043958 | 0.004975 |
| dcas_full_knn | serendipity | -0.000250 | 0.552239 |
| dcas_full_knn | cultural_calibration_kl | +0.001317 | 0.004975 |
| dcas_full_knn | minority_exposure_at_k | +0.000292 | 0.611940 |
