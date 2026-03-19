# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `E:\Desktop\Echo\reports\benchmarks\v3_main_culturemert_stage3_bprlistwise_tuned\eval\bpr_listwise_hybrid.json`
- candidate: `E:\Desktop\Echo\reports\benchmarks\v3_main_culturemert_stage3_bprlistwise_tuned\eval\dcas_full_ot_calibrated_target.json`
- paired samples: `2400`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| serendipity | 0.5135160356 | 0.8385874017 | +0.3250713661 | [0.319921, 0.329697] | 0.004975 | 2.298463 |
| cultural_calibration_kl | 1.9985830521 | 1.8792954780 | -0.1192875741 | [-0.128482, -0.110106] | 0.004975 | -0.567046 |
| minority_exposure_at_k | 0.2503333333 | 0.3814375000 | +0.1311041667 | [0.122563, 0.138943] | 0.004975 | 0.697156 |

