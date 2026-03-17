# Recommender Benchmark: v2_main_culturemert

- tracks: `E:/Desktop/Echo/storage/public/research_dataset_v2/tracks_culturemert_v2_main.npz`
- interactions: `E:\Desktop\Echo\storage\public\research_dataset_v2\interactions_v2_main.csv`
- reference_method: `dcas_full_ot`

| method | serendipity | calibration_kl | minority@k | target_prob |
|---|---:|---:|---:|---:|
| popularity | 0.600620 | 0.884802 | 0.000000 | 0.496869 |
| cosine | 0.691428 | 1.181762 | 0.344500 | 0.416574 |
| knn | 0.724240 | 1.167226 | 0.350750 | 0.419820 |
| shallow_mlp | 0.624489 | 1.342065 | 0.344000 | 0.368237 |
| hybrid_content_popularity_diversity | 0.660005 | 1.111716 | 0.171667 | 0.433642 |
| dcas_full_ot | 0.837806 | 0.805250 | 0.340083 | 0.526995 |
| dcas_full_knn | 0.837938 | 0.804816 | 0.340667 | 0.527214 |

## Reference Comparisons

| baseline | metric | delta(reference-baseline) | p_value |
|---|---|---:|---:|
| popularity | serendipity | +0.237185 | 0.003322 |
| popularity | cultural_calibration_kl | -0.079552 | 0.003322 |
| popularity | minority_exposure_at_k | +0.340083 | 0.003322 |
| cosine | serendipity | +0.146378 | 0.003322 |
| cosine | cultural_calibration_kl | -0.376512 | 0.003322 |
| cosine | minority_exposure_at_k | -0.004417 | 0.112957 |
| knn | serendipity | +0.113565 | 0.003322 |
| knn | cultural_calibration_kl | -0.361976 | 0.003322 |
| knn | minority_exposure_at_k | -0.010667 | 0.003322 |
| shallow_mlp | serendipity | +0.213317 | 0.003322 |
| shallow_mlp | cultural_calibration_kl | -0.536816 | 0.003322 |
| shallow_mlp | minority_exposure_at_k | -0.003917 | 0.182724 |
| hybrid_content_popularity_diversity | serendipity | +0.177800 | 0.003322 |
| hybrid_content_popularity_diversity | cultural_calibration_kl | -0.306466 | 0.003322 |
| hybrid_content_popularity_diversity | minority_exposure_at_k | +0.168417 | 0.003322 |
| dcas_full_knn | serendipity | -0.000132 | 0.853821 |
| dcas_full_knn | cultural_calibration_kl | +0.000434 | 0.644518 |
| dcas_full_knn | minority_exposure_at_k | -0.000583 | 0.451827 |
