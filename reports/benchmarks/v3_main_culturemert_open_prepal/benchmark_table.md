# Recommender Benchmark: v3_main_culturemert_open_prepal

- tracks: `E:/Desktop/Echo/storage/public/research_dataset_v3/tracks_culturemert_v3_main_mw3.npz`
- interactions: `E:\Desktop\Echo\storage\public\research_dataset_v3\interactions_v3_main_mixed_mw3.csv`
- reference_method: `dcas_full_ot_open`

| method | serendipity | calibration_kl | minority@k | target_prob |
|---|---:|---:|---:|---:|
| popularity | 0.454303 | 2.070947 | 0.000000 | 0.184229 |
| cosine | 0.554239 | 2.176031 | 0.226542 | 0.164475 |
| knn | 0.570462 | 2.184634 | 0.226354 | 0.162384 |
| shallow_mlp | 0.481298 | 2.130974 | 0.167396 | 0.172511 |
| hybrid_content_popularity_diversity | 0.536868 | 2.163824 | 0.070021 | 0.166311 |
| dcas_full_ot | 0.833232 | 2.069891 | 0.237000 | 0.184959 |
| dcas_full_knn | 0.834223 | 2.069716 | 0.235063 | 0.185179 |
| dcas_full_ot_open | 0.257179 | 2.295682 | 0.444354 | 0.133398 |
| dcas_full_knn_open | 0.245190 | 2.296226 | 0.446437 | 0.133378 |

## Reference Comparisons

| baseline | metric | delta(reference-baseline) | p_value |
|---|---|---:|---:|
| popularity | serendipity | -0.197123 | 0.003322 |
| popularity | cultural_calibration_kl | +0.224735 | 0.003322 |
| popularity | minority_exposure_at_k | +0.444354 | 0.003322 |
| cosine | serendipity | -0.297060 | 0.003322 |
| cosine | cultural_calibration_kl | +0.119651 | 0.003322 |
| cosine | minority_exposure_at_k | +0.217812 | 0.003322 |
| knn | serendipity | -0.313283 | 0.003322 |
| knn | cultural_calibration_kl | +0.111049 | 0.003322 |
| knn | minority_exposure_at_k | +0.218000 | 0.003322 |
| shallow_mlp | serendipity | -0.224119 | 0.003322 |
| shallow_mlp | cultural_calibration_kl | +0.164708 | 0.003322 |
| shallow_mlp | minority_exposure_at_k | +0.276958 | 0.003322 |
| hybrid_content_popularity_diversity | serendipity | -0.279689 | 0.003322 |
| hybrid_content_popularity_diversity | cultural_calibration_kl | +0.131858 | 0.003322 |
| hybrid_content_popularity_diversity | minority_exposure_at_k | +0.374333 | 0.003322 |
| dcas_full_ot | serendipity | -0.576053 | 0.003322 |
| dcas_full_ot | cultural_calibration_kl | +0.225791 | 0.003322 |
| dcas_full_ot | minority_exposure_at_k | +0.207354 | 0.003322 |
| dcas_full_knn | serendipity | -0.577044 | 0.003322 |
| dcas_full_knn | cultural_calibration_kl | +0.225966 | 0.003322 |
| dcas_full_knn | minority_exposure_at_k | +0.209292 | 0.003322 |
| dcas_full_knn_open | serendipity | +0.011989 | 0.003322 |
| dcas_full_knn_open | cultural_calibration_kl | -0.000544 | 0.385382 |
| dcas_full_knn_open | minority_exposure_at_k | -0.002083 | 0.026578 |
