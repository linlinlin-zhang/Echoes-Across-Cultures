# Recommender Benchmark: v4_main_gemini_stage3_lambdamart

- tracks: `E:/Desktop/Echo/storage/public/research_dataset_v4/main/tracks_gemini_embedding2_mw3.npz`
- interactions: `E:\Desktop\Echo\storage\public\research_dataset_v4\main\interactions_synth_mixed.csv`
- reference_method: `dcas_full_ot_calibrated_target`

| method | serendipity | calibration_kl | minority@k | target_prob |
|---|---:|---:|---:|---:|
| popularity | 0.757620 | 2.329054 | 0.000000 | 0.109537 |
| cosine | 0.851719 | 2.333392 | 0.230479 | 0.108755 |
| knn | 0.854249 | 2.333763 | 0.224562 | 0.108679 |
| lightfm_like | 0.766622 | 2.333701 | 0.116187 | 0.108631 |
| bpr_mf | 0.774738 | 2.321422 | 0.164521 | 0.111048 |
| bpr_two_stage_hybrid | 0.787001 | 2.319638 | 0.302542 | 0.111432 |
| bpr_listwise_hybrid | 0.792204 | 2.317852 | 0.286813 | 0.111787 |
| bpr_lambdamart_hybrid | 0.788364 | 2.318281 | 0.274625 | 0.111697 |
| dcas_full_ot | 0.854250 | 2.325043 | 0.195979 | 0.110348 |
| dcas_full_ot_calibrated_target | 0.824493 | 2.310406 | 0.375979 | 0.113267 |
| dcas_full_ot_calibrated_minor | 0.820860 | 2.312910 | 0.479958 | 0.112763 |

## Reference Comparisons

| baseline | metric | delta(reference-baseline) | p_value |
|---|---|---:|---:|
| popularity | serendipity | +0.066872 | 0.004975 |
| popularity | cultural_calibration_kl | -0.018648 | 0.004975 |
| popularity | minority_exposure_at_k | +0.375979 | 0.004975 |
| cosine | serendipity | -0.027227 | 0.004975 |
| cosine | cultural_calibration_kl | -0.022986 | 0.004975 |
| cosine | minority_exposure_at_k | +0.145500 | 0.004975 |
| knn | serendipity | -0.029756 | 0.004975 |
| knn | cultural_calibration_kl | -0.023357 | 0.004975 |
| knn | minority_exposure_at_k | +0.151417 | 0.004975 |
| lightfm_like | serendipity | +0.057870 | 0.004975 |
| lightfm_like | cultural_calibration_kl | -0.023295 | 0.004975 |
| lightfm_like | minority_exposure_at_k | +0.259792 | 0.004975 |
| bpr_mf | serendipity | +0.049755 | 0.004975 |
| bpr_mf | cultural_calibration_kl | -0.011016 | 0.004975 |
| bpr_mf | minority_exposure_at_k | +0.211458 | 0.004975 |
| bpr_two_stage_hybrid | serendipity | +0.037492 | 0.004975 |
| bpr_two_stage_hybrid | cultural_calibration_kl | -0.009233 | 0.004975 |
| bpr_two_stage_hybrid | minority_exposure_at_k | +0.073438 | 0.004975 |
| bpr_listwise_hybrid | serendipity | +0.032289 | 0.004975 |
| bpr_listwise_hybrid | cultural_calibration_kl | -0.007446 | 0.004975 |
| bpr_listwise_hybrid | minority_exposure_at_k | +0.089167 | 0.004975 |
| bpr_lambdamart_hybrid | serendipity | +0.036129 | 0.004975 |
| bpr_lambdamart_hybrid | cultural_calibration_kl | -0.007876 | 0.004975 |
| bpr_lambdamart_hybrid | minority_exposure_at_k | +0.101354 | 0.004975 |
| dcas_full_ot | serendipity | -0.029758 | 0.004975 |
| dcas_full_ot | cultural_calibration_kl | -0.014637 | 0.004975 |
| dcas_full_ot | minority_exposure_at_k | +0.180000 | 0.004975 |
| dcas_full_ot_calibrated_minor | serendipity | +0.003633 | 0.004975 |
| dcas_full_ot_calibrated_minor | cultural_calibration_kl | -0.002505 | 0.004975 |
| dcas_full_ot_calibrated_minor | minority_exposure_at_k | -0.103979 | 0.004975 |
