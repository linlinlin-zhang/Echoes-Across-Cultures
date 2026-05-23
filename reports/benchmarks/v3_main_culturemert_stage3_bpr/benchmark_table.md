# Recommender Benchmark: v3_main_culturemert_stage3_bpr

- tracks: `E:/Desktop/Echo/storage/public/research_dataset_v3/tracks_culturemert_v3_main_mw3.npz`
- interactions: `E:\Desktop\Echo\storage\public\research_dataset_v3\interactions_v3_main_mixed_mw3.csv`
- reference_method: `dcas_full_ot`

| method | serendipity | calibration_kl | minority@k | target_prob |
|---|---:|---:|---:|---:|
| popularity | 0.454303 | 2.070947 | 0.000000 | 0.184229 |
| cosine | 0.554239 | 2.176031 | 0.226542 | 0.164475 |
| knn | 0.570462 | 2.184634 | 0.226354 | 0.162384 |
| shallow_mlp | 0.481298 | 2.130974 | 0.167396 | 0.172511 |
| hybrid_content_popularity_diversity | 0.536868 | 2.163824 | 0.070021 | 0.166311 |
| bpr_mf | 0.491591 | 2.022577 | 0.149146 | 0.195735 |
| dcas_full_ot | 0.844854 | 2.043248 | 0.239208 | 0.191998 |
| dcas_full_knn | 0.845438 | 2.041659 | 0.239500 | 0.192516 |
| dcas_full_ot_open | 0.275580 | 2.300962 | 0.372479 | 0.132791 |
| dcas_full_knn_open | 0.263237 | 2.298760 | 0.373583 | 0.133545 |

## Reference Comparisons

| baseline | metric | delta(reference-baseline) | p_value |
|---|---|---:|---:|
| popularity | serendipity | +0.390551 | 0.003322 |
| popularity | cultural_calibration_kl | -0.027700 | 0.003322 |
| popularity | minority_exposure_at_k | +0.239208 | 0.003322 |
| cosine | serendipity | +0.290615 | 0.003322 |
| cosine | cultural_calibration_kl | -0.132783 | 0.003322 |
| cosine | minority_exposure_at_k | +0.012667 | 0.003322 |
| knn | serendipity | +0.274391 | 0.003322 |
| knn | cultural_calibration_kl | -0.141386 | 0.003322 |
| knn | minority_exposure_at_k | +0.012854 | 0.003322 |
| shallow_mlp | serendipity | +0.363555 | 0.003322 |
| shallow_mlp | cultural_calibration_kl | -0.087727 | 0.003322 |
| shallow_mlp | minority_exposure_at_k | +0.071813 | 0.003322 |
| hybrid_content_popularity_diversity | serendipity | +0.307986 | 0.003322 |
| hybrid_content_popularity_diversity | cultural_calibration_kl | -0.120577 | 0.003322 |
| hybrid_content_popularity_diversity | minority_exposure_at_k | +0.169188 | 0.003322 |
| bpr_mf | serendipity | +0.353263 | 0.003322 |
| bpr_mf | cultural_calibration_kl | +0.020671 | 0.003322 |
| bpr_mf | minority_exposure_at_k | +0.090063 | 0.003322 |
| dcas_full_knn | serendipity | -0.000584 | 0.089701 |
| dcas_full_knn | cultural_calibration_kl | +0.001589 | 0.003322 |
| dcas_full_knn | minority_exposure_at_k | -0.000292 | 0.594684 |
| dcas_full_ot_open | serendipity | +0.569274 | 0.003322 |
| dcas_full_ot_open | cultural_calibration_kl | -0.257714 | 0.003322 |
| dcas_full_ot_open | minority_exposure_at_k | -0.133271 | 0.003322 |
| dcas_full_knn_open | serendipity | +0.581616 | 0.003322 |
| dcas_full_knn_open | cultural_calibration_kl | -0.255513 | 0.003322 |
| dcas_full_knn_open | minority_exposure_at_k | -0.134375 | 0.003322 |
