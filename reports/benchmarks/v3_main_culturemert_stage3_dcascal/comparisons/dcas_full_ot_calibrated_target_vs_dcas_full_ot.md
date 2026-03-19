# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `E:\Desktop\Echo\reports\benchmarks\v3_main_culturemert_stage3_dcascal\eval\dcas_full_ot_calibrated_target.json`
- candidate: `E:\Desktop\Echo\reports\benchmarks\v3_main_culturemert_stage3_dcascal\eval\dcas_full_ot.json`
- paired samples: `2400`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| serendipity | 0.8385874017 | 0.8452264257 | +0.0066390241 | [0.004705, 0.008305] | 0.004975 | 0.145106 |
| cultural_calibration_kl | 1.8792954780 | 2.0429787993 | +0.1636833213 | [0.154142, 0.172235] | 0.004975 | 0.709010 |
| minority_exposure_at_k | 0.3814375000 | 0.2397916667 | -0.1416458333 | [-0.145989, -0.136437] | 0.004975 | -1.214257 |

