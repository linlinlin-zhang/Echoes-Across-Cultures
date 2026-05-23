# Recommender Benchmark: v3_main_gemini_embedding2

- tracks: `E:/Desktop/Echo/storage/public/research_dataset_v3/tracks_gemini_embedding2_main.npz`
- interactions: `E:\Desktop\Echo\storage\public\research_dataset_v3\interactions_v3_main.csv`
- reference_method: `dcas_full_ot`

| method | serendipity | calibration_kl | minority@k | target_prob |
|---|---:|---:|---:|---:|
| popularity | 0.805657 | 2.328180 | 0.000000 | 0.109648 |
| cosine | 0.905367 | 2.333291 | 0.391271 | 0.108737 |
| knn | 0.906172 | 2.333325 | 0.390667 | 0.108730 |
| shallow_mlp | 0.832572 | 2.335223 | 0.404104 | 0.108333 |
| hybrid_content_popularity_diversity | 0.855144 | 2.332274 | 0.175667 | 0.108926 |
| dcas_full_ot | 0.789060 | 2.327942 | 0.383604 | 0.109751 |
| dcas_full_knn | 0.788865 | 2.327995 | 0.382208 | 0.109741 |

## Reference Comparisons

| baseline | metric | delta(reference-baseline) | p_value |
|---|---|---:|---:|
| popularity | serendipity | -0.016597 | 0.003322 |
| popularity | cultural_calibration_kl | -0.000239 | 0.418605 |
| popularity | minority_exposure_at_k | +0.383604 | 0.003322 |
| cosine | serendipity | -0.116307 | 0.003322 |
| cosine | cultural_calibration_kl | -0.005350 | 0.003322 |
| cosine | minority_exposure_at_k | -0.007667 | 0.003322 |
| knn | serendipity | -0.117112 | 0.003322 |
| knn | cultural_calibration_kl | -0.005383 | 0.003322 |
| knn | minority_exposure_at_k | -0.007062 | 0.003322 |
| shallow_mlp | serendipity | -0.043512 | 0.003322 |
| shallow_mlp | cultural_calibration_kl | -0.007282 | 0.003322 |
| shallow_mlp | minority_exposure_at_k | -0.020500 | 0.003322 |
| hybrid_content_popularity_diversity | serendipity | -0.066084 | 0.003322 |
| hybrid_content_popularity_diversity | cultural_calibration_kl | -0.004332 | 0.003322 |
| hybrid_content_popularity_diversity | minority_exposure_at_k | +0.207937 | 0.003322 |
| dcas_full_knn | serendipity | +0.000195 | 0.687708 |
| dcas_full_knn | cultural_calibration_kl | -0.000053 | 0.136213 |
| dcas_full_knn | minority_exposure_at_k | +0.001396 | 0.043189 |
