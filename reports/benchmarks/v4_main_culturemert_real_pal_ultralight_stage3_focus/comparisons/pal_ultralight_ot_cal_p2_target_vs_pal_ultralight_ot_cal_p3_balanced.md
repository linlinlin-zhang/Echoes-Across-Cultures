# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `E:\Desktop\Echo\reports\benchmarks\v4_main_culturemert_real_pal_ultralight_stage3_focus\eval\pal_ultralight_ot_cal_p2_target.json`
- candidate: `E:\Desktop\Echo\reports\benchmarks\v4_main_culturemert_real_pal_ultralight_stage3_focus\eval\pal_ultralight_ot_cal_p3_balanced.json`
- paired samples: `2400`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| serendipity | 0.8350822484 | 0.8366039213 | +0.0015216730 | [0.000641, 0.002293] | 0.004975 | 0.068010 |
| cultural_calibration_kl | 2.0329659908 | 2.0325598734 | -0.0004061174 | [-0.000919, 0.000127] | 0.159204 | -0.029157 |
| minority_exposure_at_k | 0.4245416667 | 0.4668125000 | +0.0422708333 | [0.040520, 0.043689] | 0.004975 | 0.986566 |

