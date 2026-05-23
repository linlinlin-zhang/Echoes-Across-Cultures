# Recommender Benchmark: v4_main_culturemert_real_pal_ultralight_stage3_focus

- tracks: `E:/Desktop/Echo/storage/public/research_dataset_v4/main/tracks_culturemert_mw3.npz`
- interactions: `E:\Desktop\Echo\storage\public\research_dataset_v4\main\interactions_synth_mixed.csv`
- reference_method: `pal_ultralight_ot_cal_p3_balanced`

| method | serendipity | calibration_kl | minority@k | target_prob |
|---|---:|---:|---:|---:|
| pal_ultralight_ot | 0.855192 | 2.094291 | 0.244208 | 0.171371 |
| pal_ultralight_ot_cal_p2_target | 0.835082 | 2.032966 | 0.424542 | 0.185382 |
| pal_ultralight_ot_cal_p3_balanced | 0.836604 | 2.032560 | 0.466813 | 0.185625 |
| pal_ultralight_ot_cal_p5_target_minor | 0.837019 | 2.030321 | 0.481812 | 0.186127 |

## Reference Comparisons

| baseline | metric | delta(reference-baseline) | p_value |
|---|---|---:|---:|
| pal_ultralight_ot | serendipity | -0.018589 | 0.004975 |
| pal_ultralight_ot | cultural_calibration_kl | -0.061731 | 0.004975 |
| pal_ultralight_ot | minority_exposure_at_k | +0.222604 | 0.004975 |
| pal_ultralight_ot_cal_p2_target | serendipity | +0.001522 | 0.004975 |
| pal_ultralight_ot_cal_p2_target | cultural_calibration_kl | -0.000406 | 0.159204 |
| pal_ultralight_ot_cal_p2_target | minority_exposure_at_k | +0.042271 | 0.004975 |
| pal_ultralight_ot_cal_p5_target_minor | serendipity | -0.000415 | 0.059701 |
| pal_ultralight_ot_cal_p5_target_minor | cultural_calibration_kl | +0.002238 | 0.004975 |
| pal_ultralight_ot_cal_p5_target_minor | minority_exposure_at_k | -0.015000 | 0.004975 |
