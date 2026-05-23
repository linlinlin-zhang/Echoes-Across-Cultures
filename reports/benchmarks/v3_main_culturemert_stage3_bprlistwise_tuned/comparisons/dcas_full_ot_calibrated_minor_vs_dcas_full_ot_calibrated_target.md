# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `E:\Desktop\Echo\reports\benchmarks\v3_main_culturemert_stage3_bprlistwise_tuned\eval\dcas_full_ot_calibrated_minor.json`
- candidate: `E:\Desktop\Echo\reports\benchmarks\v3_main_culturemert_stage3_bprlistwise_tuned\eval\dcas_full_ot_calibrated_target.json`
- paired samples: `2400`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| serendipity | 0.8404379939 | 0.8385874017 | -0.0018505923 | [-0.003379, -0.000067] | 0.034826 | -0.046882 |
| cultural_calibration_kl | 1.9148024109 | 1.8792954780 | -0.0355069329 | [-0.038160, -0.032512] | 0.004975 | -0.496567 |
| minority_exposure_at_k | 0.5189583333 | 0.3814375000 | -0.1375208333 | [-0.141692, -0.133410] | 0.004975 | -1.319239 |

