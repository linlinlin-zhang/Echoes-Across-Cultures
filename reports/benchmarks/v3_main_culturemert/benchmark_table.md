# Recommender Benchmark: v3_main_culturemert

- tracks: `E:/Desktop/Echo/storage/public/research_dataset_v3/tracks_culturemert_v3_main.npz`
- interactions: `E:\Desktop\Echo\storage\public\research_dataset_v3\interactions_v3_main.csv`
- reference_method: `dcas_full_ot`

| method | serendipity | calibration_kl | minority@k | target_prob |
|---|---:|---:|---:|---:|
| popularity | 0.643510 | 2.040498 | 0.000000 | 0.185936 |
| cosine | 0.744366 | 2.185802 | 0.410896 | 0.160909 |
| knn | 0.768502 | 2.187656 | 0.409104 | 0.160005 |
| shallow_mlp | 0.689171 | 2.187280 | 0.401646 | 0.159106 |
| hybrid_content_popularity_diversity | 0.705608 | 2.165751 | 0.141771 | 0.164265 |
| dcas_full_ot | 0.813677 | 2.058421 | 0.402729 | 0.185575 |
| dcas_full_knn | 0.813285 | 2.058401 | 0.402313 | 0.185634 |

## Reference Comparisons

| baseline | metric | delta(reference-baseline) | p_value |
|---|---|---:|---:|
| popularity | serendipity | +0.170167 | 0.003322 |
| popularity | cultural_calibration_kl | +0.017923 | 0.003322 |
| popularity | minority_exposure_at_k | +0.402729 | 0.003322 |
| cosine | serendipity | +0.069311 | 0.003322 |
| cosine | cultural_calibration_kl | -0.127382 | 0.003322 |
| cosine | minority_exposure_at_k | -0.008167 | 0.003322 |
| knn | serendipity | +0.045175 | 0.003322 |
| knn | cultural_calibration_kl | -0.129236 | 0.003322 |
| knn | minority_exposure_at_k | -0.006375 | 0.009967 |
| shallow_mlp | serendipity | +0.124506 | 0.003322 |
| shallow_mlp | cultural_calibration_kl | -0.128860 | 0.003322 |
| shallow_mlp | minority_exposure_at_k | +0.001083 | 0.654485 |
| hybrid_content_popularity_diversity | serendipity | +0.108069 | 0.003322 |
| hybrid_content_popularity_diversity | cultural_calibration_kl | -0.107331 | 0.003322 |
| hybrid_content_popularity_diversity | minority_exposure_at_k | +0.260958 | 0.003322 |
| dcas_full_knn | serendipity | +0.000392 | 0.159468 |
| dcas_full_knn | cultural_calibration_kl | +0.000020 | 0.950166 |
| dcas_full_knn | minority_exposure_at_k | +0.000417 | 0.458472 |
