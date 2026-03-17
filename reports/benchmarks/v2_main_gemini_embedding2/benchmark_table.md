# Recommender Benchmark: v2_main_gemini_embedding2

- tracks: `E:/Desktop/Echo/storage/public/research_dataset_v2/tracks_gemini_embedding2_main.npz`
- interactions: `E:\Desktop\Echo\storage\public\research_dataset_v2\interactions_v2_main.csv`
- reference_method: `dcas_full_ot`

| method | serendipity | calibration_kl | minority@k | target_prob |
|---|---:|---:|---:|---:|
| popularity | 0.831506 | 1.759603 | 0.000000 | 0.234920 |
| cosine | 0.893858 | 1.801267 | 0.346333 | 0.225189 |
| knn | 0.897896 | 1.802071 | 0.343583 | 0.224995 |
| shallow_mlp | 0.851072 | 1.807892 | 0.354000 | 0.223491 |
| hybrid_content_popularity_diversity | 0.857469 | 1.794590 | 0.155500 | 0.226747 |
| dcas_full_ot | 0.832479 | 1.759199 | 0.361500 | 0.235109 |
| dcas_full_knn | 0.831549 | 1.759227 | 0.361000 | 0.235105 |

## Reference Comparisons

| baseline | metric | delta(reference-baseline) | p_value |
|---|---|---:|---:|
| popularity | serendipity | +0.000973 | 0.833887 |
| popularity | cultural_calibration_kl | -0.000404 | 0.760797 |
| popularity | minority_exposure_at_k | +0.361500 | 0.003322 |
| cosine | serendipity | -0.061379 | 0.003322 |
| cosine | cultural_calibration_kl | -0.042068 | 0.003322 |
| cosine | minority_exposure_at_k | +0.015167 | 0.003322 |
| knn | serendipity | -0.065417 | 0.003322 |
| knn | cultural_calibration_kl | -0.042872 | 0.003322 |
| knn | minority_exposure_at_k | +0.017917 | 0.003322 |
| shallow_mlp | serendipity | -0.018593 | 0.003322 |
| shallow_mlp | cultural_calibration_kl | -0.048693 | 0.003322 |
| shallow_mlp | minority_exposure_at_k | +0.007500 | 0.043189 |
| hybrid_content_popularity_diversity | serendipity | -0.024990 | 0.003322 |
| hybrid_content_popularity_diversity | cultural_calibration_kl | -0.035391 | 0.003322 |
| hybrid_content_popularity_diversity | minority_exposure_at_k | +0.206000 | 0.003322 |
| dcas_full_knn | serendipity | +0.000930 | 0.186047 |
| dcas_full_knn | cultural_calibration_kl | -0.000029 | 0.893688 |
| dcas_full_knn | minority_exposure_at_k | +0.000500 | 0.518272 |
