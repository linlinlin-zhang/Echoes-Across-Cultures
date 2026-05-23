# Recommender Benchmark: v4_main_culturemert_stage3_calibration_sweep

- tracks: `E:/Desktop/Echo/storage/public/research_dataset_v4/main/tracks_culturemert_mw3.npz`
- interactions: `E:\Desktop\Echo\storage\public\research_dataset_v4\main\interactions_synth_mixed.csv`
- reference_method: `dcas_ot_cal_p2_target`

| method | serendipity | calibration_kl | minority@k | target_prob |
|---|---:|---:|---:|---:|
| dcas_full_ot | 0.857861 | 2.082581 | 0.246021 | 0.174467 |
| dcas_ot_cal_p1 | 0.837089 | 2.022840 | 0.347917 | 0.189561 |
| dcas_ot_cal_p2_target | 0.831564 | 2.029638 | 0.402333 | 0.187964 |
| dcas_ot_cal_p3_balanced | 0.829856 | 2.039980 | 0.452542 | 0.185449 |
| dcas_ot_cal_p4_minor | 0.828158 | 2.047741 | 0.530271 | 0.183386 |
| dcas_ot_cal_p5_ultra_minor | 0.829623 | 2.052705 | 0.583792 | 0.182092 |

## Reference Comparisons

| baseline | metric | delta(reference-baseline) | p_value |
|---|---|---:|---:|
| dcas_full_ot | serendipity | -0.026297 | 0.004975 |
| dcas_full_ot | cultural_calibration_kl | -0.052943 | 0.004975 |
| dcas_full_ot | minority_exposure_at_k | +0.156312 | 0.004975 |
| dcas_ot_cal_p1 | serendipity | -0.005525 | 0.004975 |
| dcas_ot_cal_p1 | cultural_calibration_kl | +0.006798 | 0.004975 |
| dcas_ot_cal_p1 | minority_exposure_at_k | +0.054417 | 0.004975 |
| dcas_ot_cal_p3_balanced | serendipity | +0.001708 | 0.004975 |
| dcas_ot_cal_p3_balanced | cultural_calibration_kl | -0.010342 | 0.004975 |
| dcas_ot_cal_p3_balanced | minority_exposure_at_k | -0.050208 | 0.004975 |
| dcas_ot_cal_p4_minor | serendipity | +0.003406 | 0.004975 |
| dcas_ot_cal_p4_minor | cultural_calibration_kl | -0.018103 | 0.004975 |
| dcas_ot_cal_p4_minor | minority_exposure_at_k | -0.127938 | 0.004975 |
| dcas_ot_cal_p5_ultra_minor | serendipity | +0.001941 | 0.004975 |
| dcas_ot_cal_p5_ultra_minor | cultural_calibration_kl | -0.023067 | 0.004975 |
| dcas_ot_cal_p5_ultra_minor | minority_exposure_at_k | -0.181458 | 0.004975 |
