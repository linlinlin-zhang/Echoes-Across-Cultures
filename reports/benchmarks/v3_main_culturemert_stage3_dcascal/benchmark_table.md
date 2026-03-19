# Recommender Benchmark: v3_main_culturemert_stage3_dcascal

- tracks: `E:/Desktop/Echo/storage/public/research_dataset_v3/tracks_culturemert_v3_main_mw3.npz`
- interactions: `E:\Desktop\Echo\storage\public\research_dataset_v3\interactions_v3_main_mixed_mw3.csv`
- reference_method: `dcas_full_ot`

| method | serendipity | calibration_kl | minority@k | target_prob |
|---|---:|---:|---:|---:|
| bpr_mf | 0.491591 | 2.022577 | 0.149146 | 0.195735 |
| bpr_two_stage_hybrid | 0.510193 | 2.008183 | 0.283750 | 0.200460 |
| dcas_full_ot | 0.845226 | 2.042979 | 0.239792 | 0.192066 |
| dcas_full_ot_calibrated_target | 0.838587 | 1.879295 | 0.381437 | 0.234851 |
| dcas_full_ot_calibrated_minor | 0.840438 | 1.914802 | 0.518958 | 0.225002 |

## Reference Comparisons

| baseline | metric | delta(reference-baseline) | p_value |
|---|---|---:|---:|
| bpr_mf | serendipity | +0.353635 | 0.004975 |
| bpr_mf | cultural_calibration_kl | +0.020402 | 0.004975 |
| bpr_mf | minority_exposure_at_k | +0.090646 | 0.004975 |
| bpr_two_stage_hybrid | serendipity | +0.335034 | 0.004975 |
| bpr_two_stage_hybrid | cultural_calibration_kl | +0.034796 | 0.004975 |
| bpr_two_stage_hybrid | minority_exposure_at_k | -0.043958 | 0.004975 |
| dcas_full_ot_calibrated_target | serendipity | +0.006639 | 0.004975 |
| dcas_full_ot_calibrated_target | cultural_calibration_kl | +0.163683 | 0.004975 |
| dcas_full_ot_calibrated_target | minority_exposure_at_k | -0.141646 | 0.004975 |
| dcas_full_ot_calibrated_minor | serendipity | +0.004788 | 0.004975 |
| dcas_full_ot_calibrated_minor | cultural_calibration_kl | +0.128176 | 0.004975 |
| dcas_full_ot_calibrated_minor | minority_exposure_at_k | -0.279167 | 0.004975 |
