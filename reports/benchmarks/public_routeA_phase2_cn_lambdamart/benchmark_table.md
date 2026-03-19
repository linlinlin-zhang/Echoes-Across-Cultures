# Recommender Benchmark: public_routeA_phase2_cn_lambdamart

- tracks: `E:/Desktop/Echo/storage/public/routeA_phase2_cn/tracks.npz`
- interactions: `E:\Desktop\Echo\storage\public\routeA_phase2_cn\interactions.csv`
- reference_method: `dcas_full_ot_calibrated_target`

| method | serendipity | calibration_kl | minority@k | target_prob |
|---|---:|---:|---:|---:|
| popularity | 0.566842 | 0.996123 | 0.000000 | 0.479840 |
| cosine | 0.633852 | 1.088014 | 0.250469 | 0.454339 |
| knn | 0.668353 | 1.144311 | 0.234219 | 0.435680 |
| bpr_mf | 0.611591 | 1.009609 | 0.136406 | 0.474418 |
| bpr_two_stage_hybrid | 0.622260 | 1.071167 | 0.209375 | 0.454774 |
| bpr_listwise_hybrid | 0.637117 | 1.065354 | 0.218594 | 0.456053 |
| bpr_lambdamart_hybrid | 0.629961 | 1.001697 | 0.254688 | 0.478818 |
| dcas_full_ot | 0.765338 | 1.238358 | 0.266094 | 0.404116 |
| dcas_full_ot_calibrated_target | 0.742868 | 1.049858 | 0.374063 | 0.459604 |
| dcas_full_ot_calibrated_minor | 0.737233 | 1.102035 | 0.463438 | 0.444308 |

## Reference Comparisons

| baseline | metric | delta(reference-baseline) | p_value |
|---|---|---:|---:|
| popularity | serendipity | +0.176026 | 0.004975 |
| popularity | cultural_calibration_kl | +0.053735 | 0.004975 |
| popularity | minority_exposure_at_k | +0.374063 | 0.004975 |
| cosine | serendipity | +0.109016 | 0.004975 |
| cosine | cultural_calibration_kl | -0.038156 | 0.004975 |
| cosine | minority_exposure_at_k | +0.123594 | 0.004975 |
| knn | serendipity | +0.074515 | 0.004975 |
| knn | cultural_calibration_kl | -0.094453 | 0.004975 |
| knn | minority_exposure_at_k | +0.139844 | 0.004975 |
| bpr_mf | serendipity | +0.131277 | 0.004975 |
| bpr_mf | cultural_calibration_kl | +0.040249 | 0.004975 |
| bpr_mf | minority_exposure_at_k | +0.237656 | 0.004975 |
| bpr_two_stage_hybrid | serendipity | +0.120608 | 0.004975 |
| bpr_two_stage_hybrid | cultural_calibration_kl | -0.021309 | 0.014925 |
| bpr_two_stage_hybrid | minority_exposure_at_k | +0.164688 | 0.004975 |
| bpr_listwise_hybrid | serendipity | +0.105751 | 0.004975 |
| bpr_listwise_hybrid | cultural_calibration_kl | -0.015496 | 0.069652 |
| bpr_listwise_hybrid | minority_exposure_at_k | +0.155469 | 0.004975 |
| bpr_lambdamart_hybrid | serendipity | +0.112907 | 0.004975 |
| bpr_lambdamart_hybrid | cultural_calibration_kl | +0.048161 | 0.004975 |
| bpr_lambdamart_hybrid | minority_exposure_at_k | +0.119375 | 0.004975 |
| dcas_full_ot | serendipity | -0.022470 | 0.004975 |
| dcas_full_ot | cultural_calibration_kl | -0.188500 | 0.004975 |
| dcas_full_ot | minority_exposure_at_k | +0.107969 | 0.004975 |
| dcas_full_ot_calibrated_minor | serendipity | +0.005635 | 0.054726 |
| dcas_full_ot_calibrated_minor | cultural_calibration_kl | -0.052178 | 0.004975 |
| dcas_full_ot_calibrated_minor | minority_exposure_at_k | -0.089375 | 0.004975 |
