# Recommender Benchmark: toy_small_benchmark

- tracks: `E:/Desktop/Echo/toy_small/tracks.npz`
- interactions: `E:\Desktop\Echo\toy_small\interactions.csv`
- reference_method: `dcas_full_ot`

| method | serendipity | calibration_kl | minority@k | target_prob |
|---|---:|---:|---:|---:|
| popularity | 0.640768 | 0.042579 | 0.000000 | 0.996772 |
| cosine | 0.708092 | 0.042768 | 0.222222 | 0.992832 |
| knn | 0.739803 | 0.043027 | 0.211111 | 0.996284 |
| shallow_mlp | 0.670145 | 0.043043 | 0.211111 | 0.984155 |
| hybrid_content_popularity_diversity | 0.692806 | 0.043029 | 0.094444 | 0.993216 |
| dcas_full_ot | 0.797885 | 0.042653 | 0.127778 | 0.993676 |
| dcas_full_knn | 0.797759 | 0.042340 | 0.166667 | 0.996470 |

## Reference Comparisons

| baseline | metric | delta(reference-baseline) | p_value |
|---|---|---:|---:|
| popularity | serendipity | +0.157116 | 0.004975 |
| popularity | cultural_calibration_kl | +0.000074 | 0.935323 |
| popularity | minority_exposure_at_k | +0.127778 | 0.004975 |
| cosine | serendipity | +0.089792 | 0.034826 |
| cosine | cultural_calibration_kl | -0.000115 | 0.950249 |
| cosine | minority_exposure_at_k | -0.094444 | 0.024876 |
| knn | serendipity | +0.058082 | 0.104478 |
| knn | cultural_calibration_kl | -0.000374 | 0.800995 |
| knn | minority_exposure_at_k | -0.083333 | 0.069652 |
| shallow_mlp | serendipity | +0.127739 | 0.004975 |
| shallow_mlp | cultural_calibration_kl | -0.000389 | 0.736318 |
| shallow_mlp | minority_exposure_at_k | -0.083333 | 0.064677 |
| hybrid_content_popularity_diversity | serendipity | +0.105078 | 0.019900 |
| hybrid_content_popularity_diversity | cultural_calibration_kl | -0.000376 | 0.766169 |
| hybrid_content_popularity_diversity | minority_exposure_at_k | +0.033333 | 0.253731 |
| dcas_full_knn | serendipity | +0.000126 | 1.000000 |
| dcas_full_knn | cultural_calibration_kl | +0.000314 | 0.597015 |
| dcas_full_knn | minority_exposure_at_k | -0.038889 | 0.124378 |
