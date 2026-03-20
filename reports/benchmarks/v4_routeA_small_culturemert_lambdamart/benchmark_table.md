# Recommender Benchmark: v4_routeA_small_culturemert_lambdamart

- tracks: `E:/Desktop/Echo/storage/public/research_dataset_v4/routeA_small/tracks_culturemert_mw3.npz`
- interactions: `E:\Desktop\Echo\storage\public\research_dataset_v4\routeA_small\interactions_synth_mixed.csv`
- reference_method: `dcas_full_ot_calibrated_target`

| method | serendipity | calibration_kl | minority@k | target_prob |
|---|---:|---:|---:|---:|
| bpr_mf | 0.486381 | 1.122335 | 0.102734 | 0.429493 |
| bpr_two_stage_hybrid | 0.499763 | 1.106492 | 0.194010 | 0.436344 |
| bpr_listwise_hybrid | 0.512827 | 1.111252 | 0.200391 | 0.434319 |
| bpr_lambdamart_hybrid | 0.502178 | 1.106811 | 0.174609 | 0.435442 |
| dcas_full_ot_calibrated_target | 0.846897 | 1.087254 | 0.442708 | 0.441902 |
| dcas_full_ot_calibrated_minor | 0.837810 | 1.116872 | 0.654036 | 0.431507 |

## Reference Comparisons

| baseline | metric | delta(reference-baseline) | p_value |
|---|---|---:|---:|
| bpr_mf | serendipity | +0.360516 | 0.004975 |
| bpr_mf | cultural_calibration_kl | -0.035081 | 0.004975 |
| bpr_mf | minority_exposure_at_k | +0.339974 | 0.004975 |
| bpr_two_stage_hybrid | serendipity | +0.347134 | 0.004975 |
| bpr_two_stage_hybrid | cultural_calibration_kl | -0.019239 | 0.004975 |
| bpr_two_stage_hybrid | minority_exposure_at_k | +0.248698 | 0.004975 |
| bpr_listwise_hybrid | serendipity | +0.334070 | 0.004975 |
| bpr_listwise_hybrid | cultural_calibration_kl | -0.023999 | 0.004975 |
| bpr_listwise_hybrid | minority_exposure_at_k | +0.242318 | 0.004975 |
| bpr_lambdamart_hybrid | serendipity | +0.344719 | 0.004975 |
| bpr_lambdamart_hybrid | cultural_calibration_kl | -0.019558 | 0.004975 |
| bpr_lambdamart_hybrid | minority_exposure_at_k | +0.268099 | 0.004975 |
| dcas_full_ot_calibrated_minor | serendipity | +0.009087 | 0.004975 |
| dcas_full_ot_calibrated_minor | cultural_calibration_kl | -0.029619 | 0.004975 |
| dcas_full_ot_calibrated_minor | minority_exposure_at_k | -0.211328 | 0.004975 |
