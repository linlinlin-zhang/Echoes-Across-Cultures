# Recommender Benchmark: v3_main_culturemert_stage3_bprlistwise

- tracks: `E:/Desktop/Echo/storage/public/research_dataset_v3/tracks_culturemert_v3_main_mw3.npz`
- interactions: `E:\Desktop\Echo\storage\public\research_dataset_v3\interactions_v3_main_mixed_mw3.csv`
- reference_method: `dcas_full_ot_calibrated_target`

| method | serendipity | calibration_kl | minority@k | target_prob |
|---|---:|---:|---:|---:|
| bpr_mf | 0.491591 | 2.022577 | 0.149146 | 0.195735 |
| bpr_two_stage_hybrid | 0.510193 | 2.008183 | 0.283750 | 0.200460 |
| bpr_listwise_hybrid | 0.506162 | 2.010272 | 0.225083 | 0.199415 |
| dcas_full_ot_calibrated_target | 0.838587 | 1.879295 | 0.381437 | 0.234851 |
| dcas_full_ot_calibrated_minor | 0.840438 | 1.914802 | 0.518958 | 0.225002 |

## Reference Comparisons

| baseline | metric | delta(reference-baseline) | p_value |
|---|---|---:|---:|
| bpr_mf | serendipity | +0.346996 | 0.004975 |
| bpr_mf | cultural_calibration_kl | -0.143281 | 0.004975 |
| bpr_mf | minority_exposure_at_k | +0.232292 | 0.004975 |
| bpr_two_stage_hybrid | serendipity | +0.328395 | 0.004975 |
| bpr_two_stage_hybrid | cultural_calibration_kl | -0.128887 | 0.004975 |
| bpr_two_stage_hybrid | minority_exposure_at_k | +0.097688 | 0.004975 |
| bpr_listwise_hybrid | serendipity | +0.332426 | 0.004975 |
| bpr_listwise_hybrid | cultural_calibration_kl | -0.130977 | 0.004975 |
| bpr_listwise_hybrid | minority_exposure_at_k | +0.156354 | 0.004975 |
| dcas_full_ot_calibrated_minor | serendipity | -0.001851 | 0.034826 |
| dcas_full_ot_calibrated_minor | cultural_calibration_kl | -0.035507 | 0.004975 |
| dcas_full_ot_calibrated_minor | minority_exposure_at_k | -0.137521 | 0.004975 |
