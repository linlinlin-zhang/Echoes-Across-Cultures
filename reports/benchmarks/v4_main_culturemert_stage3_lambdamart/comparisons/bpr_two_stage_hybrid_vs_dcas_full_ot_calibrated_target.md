# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `E:\Desktop\Echo\reports\benchmarks\v4_main_culturemert_stage3_lambdamart\eval\bpr_two_stage_hybrid.json`
- candidate: `E:\Desktop\Echo\reports\benchmarks\v4_main_culturemert_stage3_lambdamart\eval\dcas_full_ot_calibrated_target.json`
- paired samples: `2400`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| serendipity | 0.5680538343 | 0.8315641066 | +0.2635102723 | [0.257592, 0.269044] | 0.000999 | 1.786181 |
| cultural_calibration_kl | 2.1044863123 | 2.0296378083 | -0.0748485040 | [-0.080417, -0.069724] | 0.000999 | -0.555898 |
| minority_exposure_at_k | 0.2943333333 | 0.4023333333 | +0.1080000000 | [0.101124, 0.114626] | 0.000999 | 0.639854 |

