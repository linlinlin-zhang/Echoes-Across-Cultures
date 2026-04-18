# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `reports\benchmarks\v4_main_culturemert_stage3_lambdamart\eval\dcas_full_ot_calibrated_target.json`
- candidate: `reports\benchmarks\v4_main_culturemert_real_pal_ultralight_stage3_focus\eval\pal_ultralight_ot_cal_p3_balanced.json`
- paired samples: `2400`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| serendipity | 0.8315641066 | 0.8366039213 | +0.0050398147 | [0.003199, 0.006984] | 0.003322 | 0.099661 |
| cultural_calibration_kl | 2.0296378083 | 2.0325598734 | +0.0029220652 | [0.001330, 0.004496] | 0.003322 | 0.073468 |
| minority_exposure_at_k | 0.4023333333 | 0.4668125000 | +0.0644791667 | [0.060967, 0.067981] | 0.003322 | 0.709577 |
| target_culture_prob_mean | 0.1879636552 | 0.1856246550 | -0.0023390002 | [-0.002776, -0.001892] | 0.003322 | -0.203517 |

