# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `E:\Desktop\Echo\reports\benchmarks\v4_main_culturemert_real_pal_ultralight_stage3_focus\eval\pal_ultralight_ot_cal_p5_target_minor.json`
- candidate: `E:\Desktop\Echo\reports\benchmarks\v4_main_culturemert_real_pal_ultralight_stage3_focus\eval\pal_ultralight_ot_cal_p3_balanced.json`
- paired samples: `2400`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| serendipity | 0.8370190908 | 0.8366039213 | -0.0004151695 | [-0.000854, -0.000014] | 0.059701 | -0.037840 |
| cultural_calibration_kl | 2.0303214609 | 2.0325598734 | +0.0022384125 | [0.001816, 0.002698] | 0.004975 | 0.219216 |
| minority_exposure_at_k | 0.4818125000 | 0.4668125000 | -0.0150000000 | [-0.016125, -0.013792] | 0.004975 | -0.486030 |

