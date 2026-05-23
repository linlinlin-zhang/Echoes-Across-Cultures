# Recommender Benchmark: v4_routeA_small_gemini_stage3_lambdamart

- tracks: `E:/Desktop/Echo/storage/public/research_dataset_v4/routeA_small/tracks_gemini_embedding2_mw3.npz`
- interactions: `E:\Desktop\Echo\storage\public\research_dataset_v4\routeA_small\interactions_synth_mixed.csv`
- reference_method: `dcas_full_ot_calibrated_target`

| method | serendipity | calibration_kl | minority@k | target_prob |
|---|---:|---:|---:|---:|
| popularity | 0.660338 | 1.580523 | 0.000000 | 0.279544 |
| cosine | 0.790129 | 1.581755 | 0.191406 | 0.279284 |
| knn | 0.788096 | 1.580136 | 0.184245 | 0.279701 |
| lightfm_like | 0.684419 | 1.577308 | 0.046094 | 0.280381 |
| bpr_mf | 0.710395 | 1.572263 | 0.095833 | 0.281650 |
| bpr_two_stage_hybrid | 0.727958 | 1.568499 | 0.145182 | 0.282626 |
| bpr_listwise_hybrid | 0.733121 | 1.566272 | 0.165625 | 0.283203 |
| bpr_lambdamart_hybrid | 0.726665 | 1.567630 | 0.151302 | 0.282851 |
| dcas_full_ot | 0.860568 | 1.572332 | 0.242318 | 0.281668 |
| dcas_full_ot_calibrated_target | 0.864198 | 1.550196 | 0.499740 | 0.287335 |
| dcas_full_ot_calibrated_minor | 0.858572 | 1.557353 | 0.652865 | 0.285482 |

## Reference Comparisons

| baseline | metric | delta(reference-baseline) | p_value |
|---|---|---:|---:|
| popularity | serendipity | +0.203860 | 0.004975 |
| popularity | cultural_calibration_kl | -0.030326 | 0.004975 |
| popularity | minority_exposure_at_k | +0.499740 | 0.004975 |
| cosine | serendipity | +0.074069 | 0.004975 |
| cosine | cultural_calibration_kl | -0.031559 | 0.004975 |
| cosine | minority_exposure_at_k | +0.308333 | 0.004975 |
| knn | serendipity | +0.076101 | 0.004975 |
| knn | cultural_calibration_kl | -0.029940 | 0.004975 |
| knn | minority_exposure_at_k | +0.315495 | 0.004975 |
| lightfm_like | serendipity | +0.179779 | 0.004975 |
| lightfm_like | cultural_calibration_kl | -0.027112 | 0.004975 |
| lightfm_like | minority_exposure_at_k | +0.453646 | 0.004975 |
| bpr_mf | serendipity | +0.153802 | 0.004975 |
| bpr_mf | cultural_calibration_kl | -0.022067 | 0.004975 |
| bpr_mf | minority_exposure_at_k | +0.403906 | 0.004975 |
| bpr_two_stage_hybrid | serendipity | +0.136240 | 0.004975 |
| bpr_two_stage_hybrid | cultural_calibration_kl | -0.018302 | 0.004975 |
| bpr_two_stage_hybrid | minority_exposure_at_k | +0.354557 | 0.004975 |
| bpr_listwise_hybrid | serendipity | +0.131076 | 0.004975 |
| bpr_listwise_hybrid | cultural_calibration_kl | -0.016075 | 0.004975 |
| bpr_listwise_hybrid | minority_exposure_at_k | +0.334115 | 0.004975 |
| bpr_lambdamart_hybrid | serendipity | +0.137533 | 0.004975 |
| bpr_lambdamart_hybrid | cultural_calibration_kl | -0.017434 | 0.004975 |
| bpr_lambdamart_hybrid | minority_exposure_at_k | +0.348437 | 0.004975 |
| dcas_full_ot | serendipity | +0.003630 | 0.248756 |
| dcas_full_ot | cultural_calibration_kl | -0.022136 | 0.004975 |
| dcas_full_ot | minority_exposure_at_k | +0.257422 | 0.004975 |
| dcas_full_ot_calibrated_minor | serendipity | +0.005626 | 0.004975 |
| dcas_full_ot_calibrated_minor | cultural_calibration_kl | -0.007157 | 0.004975 |
| dcas_full_ot_calibrated_minor | minority_exposure_at_k | -0.153125 | 0.004975 |
