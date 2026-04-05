# Recommender Benchmark: v4_routeA_small_gemini_stage3_calibration_sweep

- tracks: `E:/Desktop/Echo/storage/public/research_dataset_v4/routeA_small/tracks_gemini_embedding2_mw3.npz`
- interactions: `E:\Desktop\Echo\storage\public\research_dataset_v4\routeA_small\interactions_synth_mixed.csv`
- reference_method: `dcas_ot_cal_p2_target`

| method | serendipity | calibration_kl | minority@k | target_prob |
|---|---:|---:|---:|---:|
| dcas_full_ot | 0.860568 | 1.572332 | 0.242318 | 0.281668 |
| dcas_ot_cal_p1 | 0.866729 | 1.548636 | 0.435937 | 0.287740 |
| dcas_ot_cal_p2_target | 0.864198 | 1.550196 | 0.499740 | 0.287335 |
| dcas_ot_cal_p3_balanced | 0.861987 | 1.553454 | 0.568620 | 0.286490 |
| dcas_ot_cal_p4_minor | 0.858572 | 1.557353 | 0.652865 | 0.285482 |
| dcas_ot_cal_p5_ultra_minor | 0.856613 | 1.559496 | 0.708464 | 0.284929 |

## Reference Comparisons

| baseline | metric | delta(reference-baseline) | p_value |
|---|---|---:|---:|
| dcas_full_ot | serendipity | +0.003630 | 0.248756 |
| dcas_full_ot | cultural_calibration_kl | -0.022136 | 0.004975 |
| dcas_full_ot | minority_exposure_at_k | +0.257422 | 0.004975 |
| dcas_ot_cal_p1 | serendipity | -0.002531 | 0.039801 |
| dcas_ot_cal_p1 | cultural_calibration_kl | +0.001561 | 0.004975 |
| dcas_ot_cal_p1 | minority_exposure_at_k | +0.063802 | 0.004975 |
| dcas_ot_cal_p3_balanced | serendipity | +0.002210 | 0.024876 |
| dcas_ot_cal_p3_balanced | cultural_calibration_kl | -0.003258 | 0.004975 |
| dcas_ot_cal_p3_balanced | minority_exposure_at_k | -0.068880 | 0.004975 |
| dcas_ot_cal_p4_minor | serendipity | +0.005626 | 0.004975 |
| dcas_ot_cal_p4_minor | cultural_calibration_kl | -0.007157 | 0.004975 |
| dcas_ot_cal_p4_minor | minority_exposure_at_k | -0.153125 | 0.004975 |
| dcas_ot_cal_p5_ultra_minor | serendipity | +0.007584 | 0.004975 |
| dcas_ot_cal_p5_ultra_minor | cultural_calibration_kl | -0.009300 | 0.004975 |
| dcas_ot_cal_p5_ultra_minor | minority_exposure_at_k | -0.208724 | 0.004975 |
