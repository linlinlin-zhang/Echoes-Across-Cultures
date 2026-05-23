# Recommender Benchmark: v4_main_culturemert_lambdamart

- tracks: `E:/Desktop/Echo/storage/public/research_dataset_v4/main/tracks_culturemert_mw3.npz`
- interactions: `E:\Desktop\Echo\storage\public\research_dataset_v4\main\interactions_synth_mixed_culturemert_mw3.csv`
- reference_method: `dcas_full_ot_calibrated_target`

| method | serendipity | calibration_kl | minority@k | target_prob |
|---|---:|---:|---:|---:|
| bpr_mf | 0.537262 | 2.113856 | 0.161542 | 0.165720 |
| bpr_two_stage_hybrid | 0.560509 | 2.100527 | 0.265271 | 0.169364 |
| bpr_listwise_hybrid | 0.567105 | 2.096395 | 0.277500 | 0.170719 |
| bpr_lambdamart_hybrid | 0.559368 | 2.093687 | 0.272313 | 0.171217 |
| dcas_full_ot_calibrated_target | 0.811056 | 2.049207 | 0.399813 | 0.183208 |
| dcas_full_ot_calibrated_minor | 0.807592 | 2.061880 | 0.506062 | 0.179897 |

## Reference Comparisons

| baseline | metric | delta(reference-baseline) | p_value |
|---|---|---:|---:|
| bpr_mf | serendipity | +0.273794 | 0.004975 |
| bpr_mf | cultural_calibration_kl | -0.064648 | 0.004975 |
| bpr_mf | minority_exposure_at_k | +0.238271 | 0.004975 |
| bpr_two_stage_hybrid | serendipity | +0.250546 | 0.004975 |
| bpr_two_stage_hybrid | cultural_calibration_kl | -0.051320 | 0.004975 |
| bpr_two_stage_hybrid | minority_exposure_at_k | +0.134542 | 0.004975 |
| bpr_listwise_hybrid | serendipity | +0.243951 | 0.004975 |
| bpr_listwise_hybrid | cultural_calibration_kl | -0.047187 | 0.004975 |
| bpr_listwise_hybrid | minority_exposure_at_k | +0.122313 | 0.004975 |
| bpr_lambdamart_hybrid | serendipity | +0.251688 | 0.004975 |
| bpr_lambdamart_hybrid | cultural_calibration_kl | -0.044479 | 0.004975 |
| bpr_lambdamart_hybrid | minority_exposure_at_k | +0.127500 | 0.004975 |
| dcas_full_ot_calibrated_minor | serendipity | +0.003464 | 0.004975 |
| dcas_full_ot_calibrated_minor | cultural_calibration_kl | -0.012672 | 0.004975 |
| dcas_full_ot_calibrated_minor | minority_exposure_at_k | -0.106250 | 0.004975 |
