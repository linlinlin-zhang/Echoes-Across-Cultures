# Recommender Benchmark: v4_routeA_small_culturemert_stage3_lambdamart

- tracks: `E:/Desktop/Echo/storage/public/research_dataset_v4/routeA_small/tracks_culturemert_mw3.npz`
- interactions: `E:\Desktop\Echo\storage\public\research_dataset_v4\routeA_small\interactions_synth_mixed.csv`
- reference_method: `dcas_full_ot_calibrated_target`

| method | serendipity | calibration_kl | minority@k | target_prob |
|---|---:|---:|---:|---:|
| popularity | 0.386587 | 1.120704 | 0.000000 | 0.428283 |
| cosine | 0.430372 | 1.160764 | 0.212109 | 0.421012 |
| knn | 0.445182 | 1.174174 | 0.242448 | 0.416954 |
| lightfm_like | 0.384640 | 1.148551 | 0.119401 | 0.421706 |
| bpr_mf | 0.481093 | 1.129262 | 0.087891 | 0.427195 |
| bpr_two_stage_hybrid | 0.500245 | 1.126419 | 0.178776 | 0.429204 |
| bpr_listwise_hybrid | 0.504103 | 1.119935 | 0.192188 | 0.431334 |
| bpr_lambdamart_hybrid | 0.501788 | 1.113629 | 0.157292 | 0.433312 |
| dcas_full_ot | 0.850135 | 1.143915 | 0.302734 | 0.424368 |
| dcas_full_ot_calibrated_target | 0.840630 | 1.021299 | 0.509505 | 0.463446 |
| dcas_full_ot_calibrated_minor | 0.837080 | 1.074745 | 0.679818 | 0.445691 |

## Reference Comparisons

| baseline | metric | delta(reference-baseline) | p_value |
|---|---|---:|---:|
| popularity | serendipity | +0.454043 | 0.004975 |
| popularity | cultural_calibration_kl | -0.099405 | 0.004975 |
| popularity | minority_exposure_at_k | +0.509505 | 0.004975 |
| cosine | serendipity | +0.410258 | 0.004975 |
| cosine | cultural_calibration_kl | -0.139465 | 0.004975 |
| cosine | minority_exposure_at_k | +0.297396 | 0.004975 |
| knn | serendipity | +0.395448 | 0.004975 |
| knn | cultural_calibration_kl | -0.152875 | 0.004975 |
| knn | minority_exposure_at_k | +0.267057 | 0.004975 |
| lightfm_like | serendipity | +0.455990 | 0.004975 |
| lightfm_like | cultural_calibration_kl | -0.127252 | 0.004975 |
| lightfm_like | minority_exposure_at_k | +0.390104 | 0.004975 |
| bpr_mf | serendipity | +0.359536 | 0.004975 |
| bpr_mf | cultural_calibration_kl | -0.107963 | 0.004975 |
| bpr_mf | minority_exposure_at_k | +0.421615 | 0.004975 |
| bpr_two_stage_hybrid | serendipity | +0.340385 | 0.004975 |
| bpr_two_stage_hybrid | cultural_calibration_kl | -0.105120 | 0.004975 |
| bpr_two_stage_hybrid | minority_exposure_at_k | +0.330729 | 0.004975 |
| bpr_listwise_hybrid | serendipity | +0.336527 | 0.004975 |
| bpr_listwise_hybrid | cultural_calibration_kl | -0.098636 | 0.004975 |
| bpr_listwise_hybrid | minority_exposure_at_k | +0.317318 | 0.004975 |
| bpr_lambdamart_hybrid | serendipity | +0.338842 | 0.004975 |
| bpr_lambdamart_hybrid | cultural_calibration_kl | -0.092330 | 0.004975 |
| bpr_lambdamart_hybrid | minority_exposure_at_k | +0.352214 | 0.004975 |
| dcas_full_ot | serendipity | -0.009505 | 0.004975 |
| dcas_full_ot | cultural_calibration_kl | -0.122616 | 0.004975 |
| dcas_full_ot | minority_exposure_at_k | +0.206771 | 0.004975 |
| dcas_full_ot_calibrated_minor | serendipity | +0.003550 | 0.079602 |
| dcas_full_ot_calibrated_minor | cultural_calibration_kl | -0.053446 | 0.004975 |
| dcas_full_ot_calibrated_minor | minority_exposure_at_k | -0.170313 | 0.004975 |
