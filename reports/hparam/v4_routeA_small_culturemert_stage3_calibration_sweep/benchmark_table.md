# Recommender Benchmark: v4_routeA_small_culturemert_stage3_calibration_sweep

- tracks: `E:/Desktop/Echo/storage/public/research_dataset_v4/routeA_small/tracks_culturemert_mw3.npz`
- interactions: `E:\Desktop\Echo\storage\public\research_dataset_v4\routeA_small\interactions_synth_mixed.csv`
- reference_method: `dcas_ot_cal_p2_target`

| method | serendipity | calibration_kl | minority@k | target_prob |
|---|---:|---:|---:|---:|
| dcas_full_ot | 0.850135 | 1.143915 | 0.302734 | 0.424368 |
| dcas_ot_cal_p1 | 0.837797 | 1.013613 | 0.437370 | 0.466029 |
| dcas_ot_cal_p2_target | 0.840630 | 1.021299 | 0.509505 | 0.463446 |
| dcas_ot_cal_p3_balanced | 0.838404 | 1.043575 | 0.579427 | 0.456188 |
| dcas_ot_cal_p4_minor | 0.837080 | 1.074745 | 0.679818 | 0.445691 |
| dcas_ot_cal_p5_ultra_minor | 0.838454 | 1.089537 | 0.752865 | 0.440684 |

## Reference Comparisons

| baseline | metric | delta(reference-baseline) | p_value |
|---|---|---:|---:|
| dcas_full_ot | serendipity | -0.009505 | 0.004975 |
| dcas_full_ot | cultural_calibration_kl | -0.122616 | 0.004975 |
| dcas_full_ot | minority_exposure_at_k | +0.206771 | 0.004975 |
| dcas_ot_cal_p1 | serendipity | +0.002833 | 0.019900 |
| dcas_ot_cal_p1 | cultural_calibration_kl | +0.007686 | 0.004975 |
| dcas_ot_cal_p1 | minority_exposure_at_k | +0.072135 | 0.004975 |
| dcas_ot_cal_p3_balanced | serendipity | +0.002226 | 0.084577 |
| dcas_ot_cal_p3_balanced | cultural_calibration_kl | -0.022276 | 0.004975 |
| dcas_ot_cal_p3_balanced | minority_exposure_at_k | -0.069922 | 0.004975 |
| dcas_ot_cal_p4_minor | serendipity | +0.003550 | 0.079602 |
| dcas_ot_cal_p4_minor | cultural_calibration_kl | -0.053446 | 0.004975 |
| dcas_ot_cal_p4_minor | minority_exposure_at_k | -0.170313 | 0.004975 |
| dcas_ot_cal_p5_ultra_minor | serendipity | +0.002176 | 0.328358 |
| dcas_ot_cal_p5_ultra_minor | cultural_calibration_kl | -0.068238 | 0.004975 |
| dcas_ot_cal_p5_ultra_minor | minority_exposure_at_k | -0.243359 | 0.004975 |
