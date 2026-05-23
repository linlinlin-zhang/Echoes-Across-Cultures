# Recommender Benchmark: v4_main_culturemert_real_pal_stage3_calibration_sweep

- tracks: `E:/Desktop/Echo/storage/public/research_dataset_v4/main/tracks_culturemert_mw3.npz`
- interactions: `E:\Desktop\Echo\storage\public\research_dataset_v4\main\interactions_synth_mixed.csv`
- reference_method: `pal_ot_cal_p2_target`

| method | serendipity | calibration_kl | minority@k | target_prob |
|---|---:|---:|---:|---:|
| pal_ot | 0.859624 | 2.059524 | 0.245312 | 0.180560 |
| pal_ot_cal_p1 | 0.841078 | 2.013783 | 0.334833 | 0.191371 |
| pal_ot_cal_p2_target | 0.836952 | 2.019681 | 0.386563 | 0.190022 |
| pal_ot_cal_p3_balanced | 0.837654 | 2.021153 | 0.441500 | 0.189956 |
| pal_ot_cal_p4_minor | 0.834354 | 2.031867 | 0.507708 | 0.187423 |
| pal_ot_cal_p5_target_minor | 0.834656 | 2.019806 | 0.463917 | 0.190370 |

## Reference Comparisons

| baseline | metric | delta(reference-baseline) | p_value |
|---|---|---:|---:|
| pal_ot | serendipity | -0.022672 | 0.004975 |
| pal_ot | cultural_calibration_kl | -0.039843 | 0.004975 |
| pal_ot | minority_exposure_at_k | +0.141250 | 0.004975 |
| pal_ot_cal_p1 | serendipity | -0.004126 | 0.004975 |
| pal_ot_cal_p1 | cultural_calibration_kl | +0.005898 | 0.004975 |
| pal_ot_cal_p1 | minority_exposure_at_k | +0.051729 | 0.004975 |
| pal_ot_cal_p3_balanced | serendipity | -0.000702 | 0.208955 |
| pal_ot_cal_p3_balanced | cultural_calibration_kl | -0.001472 | 0.004975 |
| pal_ot_cal_p3_balanced | minority_exposure_at_k | -0.054938 | 0.004975 |
| pal_ot_cal_p4_minor | serendipity | +0.002598 | 0.004975 |
| pal_ot_cal_p4_minor | cultural_calibration_kl | -0.012186 | 0.004975 |
| pal_ot_cal_p4_minor | minority_exposure_at_k | -0.121146 | 0.004975 |
| pal_ot_cal_p5_target_minor | serendipity | +0.002296 | 0.004975 |
| pal_ot_cal_p5_target_minor | cultural_calibration_kl | -0.000124 | 0.751244 |
| pal_ot_cal_p5_target_minor | minority_exposure_at_k | -0.077354 | 0.004975 |
