# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `E:\Desktop\Echo\reports\benchmarks\v3_main_culturemert_stage3_bprlistwise\eval\bpr_two_stage_hybrid.json`
- candidate: `E:\Desktop\Echo\reports\benchmarks\v3_main_culturemert_stage3_bprlistwise\eval\dcas_full_ot_calibrated_target.json`
- paired samples: `2400`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| serendipity | 0.5101928902 | 0.8385874017 | +0.3283945114 | [0.323602, 0.333426] | 0.004975 | 2.331398 |
| cultural_calibration_kl | 2.0081828249 | 1.8792954780 | -0.1288873469 | [-0.137499, -0.120042] | 0.004975 | -0.641550 |
| minority_exposure_at_k | 0.2837500000 | 0.3814375000 | +0.0976875000 | [0.090074, 0.104446] | 0.004975 | 0.567540 |

