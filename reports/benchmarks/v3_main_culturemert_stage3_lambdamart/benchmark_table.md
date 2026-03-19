# Recommender Benchmark: v3_main_culturemert_stage3_lambdamart

- tracks: `E:/Desktop/Echo/storage/public/research_dataset_v3/tracks_culturemert_v3_main_mw3.npz`
- interactions: `E:\Desktop\Echo\storage\public\research_dataset_v3\interactions_v3_main_mixed_mw3.csv`
- reference_method: `dcas_full_ot_calibrated_target`

| method | serendipity | calibration_kl | minority@k | target_prob |
|---|---:|---:|---:|---:|
| bpr_mf | 0.491591 | 2.022577 | 0.149146 | 0.195735 |
| bpr_two_stage_hybrid | 0.510193 | 2.008183 | 0.283750 | 0.200460 |
| bpr_listwise_hybrid | 0.513516 | 1.998583 | 0.250333 | 0.202936 |
| bpr_lambdamart_hybrid | 0.510684 | 1.996740 | 0.266208 | 0.203780 |
| dcas_full_ot_calibrated_target | 0.838588 | 1.879295 | 0.381437 | 0.234851 |
| dcas_full_ot_calibrated_minor | 0.840440 | 1.914802 | 0.518979 | 0.225002 |

## Reference Comparisons

| baseline | metric | delta(reference-baseline) | p_value |
|---|---|---:|---:|
| bpr_mf | serendipity | +0.346996 | 0.004975 |
| bpr_mf | cultural_calibration_kl | -0.143282 | 0.004975 |
| bpr_mf | minority_exposure_at_k | +0.232292 | 0.004975 |
| bpr_two_stage_hybrid | serendipity | +0.328395 | 0.004975 |
| bpr_two_stage_hybrid | cultural_calibration_kl | -0.128888 | 0.004975 |
| bpr_two_stage_hybrid | minority_exposure_at_k | +0.097688 | 0.004975 |
| bpr_listwise_hybrid | serendipity | +0.325072 | 0.004975 |
| bpr_listwise_hybrid | cultural_calibration_kl | -0.119288 | 0.004975 |
| bpr_listwise_hybrid | minority_exposure_at_k | +0.131104 | 0.004975 |
| bpr_lambdamart_hybrid | serendipity | +0.327903 | 0.004975 |
| bpr_lambdamart_hybrid | cultural_calibration_kl | -0.117446 | 0.004975 |
| bpr_lambdamart_hybrid | minority_exposure_at_k | +0.115229 | 0.004975 |
| dcas_full_ot_calibrated_minor | serendipity | -0.001852 | 0.034826 |
| dcas_full_ot_calibrated_minor | cultural_calibration_kl | -0.035507 | 0.004975 |
| dcas_full_ot_calibrated_minor | minority_exposure_at_k | -0.137542 | 0.004975 |
