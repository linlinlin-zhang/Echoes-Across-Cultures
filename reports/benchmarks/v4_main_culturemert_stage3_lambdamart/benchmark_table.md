# Recommender Benchmark: v4_main_culturemert_stage3_lambdamart

- tracks: `E:/Desktop/Echo/storage/public/research_dataset_v4/main/tracks_culturemert_mw3.npz`
- interactions: `E:\Desktop\Echo\storage\public\research_dataset_v4\main\interactions_synth_mixed.csv`
- reference_method: `dcas_full_ot_calibrated_target`

| method | serendipity | calibration_kl | minority@k | target_prob |
|---|---:|---:|---:|---:|
| popularity | 0.501464 | 2.173411 | 0.000000 | 0.151525 |
| cosine | 0.633324 | 2.233377 | 0.220667 | 0.141885 |
| knn | 0.644522 | 2.234130 | 0.213063 | 0.141954 |
| lightfm_like | 0.502662 | 2.185023 | 0.133625 | 0.149577 |
| bpr_mf | 0.537290 | 2.114678 | 0.164625 | 0.165561 |
| bpr_two_stage_hybrid | 0.568054 | 2.104486 | 0.294333 | 0.168815 |
| bpr_listwise_hybrid | 0.561400 | 2.095753 | 0.278292 | 0.170691 |
| bpr_lambdamart_hybrid | 0.555827 | 2.096581 | 0.268083 | 0.170585 |
| dcas_full_ot | 0.857861 | 2.082581 | 0.246021 | 0.174467 |
| dcas_full_ot_calibrated_target | 0.831564 | 2.029638 | 0.402333 | 0.187964 |
| dcas_full_ot_calibrated_minor | 0.828158 | 2.047741 | 0.530271 | 0.183386 |

## Reference Comparisons

| baseline | metric | delta(reference-baseline) | p_value |
|---|---|---:|---:|
| popularity | serendipity | +0.330100 | 0.004975 |
| popularity | cultural_calibration_kl | -0.143773 | 0.004975 |
| popularity | minority_exposure_at_k | +0.402333 | 0.004975 |
| cosine | serendipity | +0.198240 | 0.004975 |
| cosine | cultural_calibration_kl | -0.203739 | 0.004975 |
| cosine | minority_exposure_at_k | +0.181667 | 0.004975 |
| knn | serendipity | +0.187042 | 0.004975 |
| knn | cultural_calibration_kl | -0.204493 | 0.004975 |
| knn | minority_exposure_at_k | +0.189271 | 0.004975 |
| lightfm_like | serendipity | +0.328902 | 0.004975 |
| lightfm_like | cultural_calibration_kl | -0.155385 | 0.004975 |
| lightfm_like | minority_exposure_at_k | +0.268708 | 0.004975 |
| bpr_mf | serendipity | +0.294274 | 0.004975 |
| bpr_mf | cultural_calibration_kl | -0.085040 | 0.004975 |
| bpr_mf | minority_exposure_at_k | +0.237708 | 0.004975 |
| bpr_two_stage_hybrid | serendipity | +0.263510 | 0.004975 |
| bpr_two_stage_hybrid | cultural_calibration_kl | -0.074849 | 0.004975 |
| bpr_two_stage_hybrid | minority_exposure_at_k | +0.108000 | 0.004975 |
| bpr_listwise_hybrid | serendipity | +0.270164 | 0.004975 |
| bpr_listwise_hybrid | cultural_calibration_kl | -0.066115 | 0.004975 |
| bpr_listwise_hybrid | minority_exposure_at_k | +0.124042 | 0.004975 |
| bpr_lambdamart_hybrid | serendipity | +0.275737 | 0.004975 |
| bpr_lambdamart_hybrid | cultural_calibration_kl | -0.066943 | 0.004975 |
| bpr_lambdamart_hybrid | minority_exposure_at_k | +0.134250 | 0.004975 |
| dcas_full_ot | serendipity | -0.026297 | 0.004975 |
| dcas_full_ot | cultural_calibration_kl | -0.052943 | 0.004975 |
| dcas_full_ot | minority_exposure_at_k | +0.156312 | 0.004975 |
| dcas_full_ot_calibrated_minor | serendipity | +0.003406 | 0.004975 |
| dcas_full_ot_calibrated_minor | cultural_calibration_kl | -0.018103 | 0.004975 |
| dcas_full_ot_calibrated_minor | minority_exposure_at_k | -0.127938 | 0.004975 |
