# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `E:\Desktop\Echo\reports\benchmarks\v4_main_culturemert_stage3_lambdamart\eval\bpr_listwise_hybrid.json`
- candidate: `E:\Desktop\Echo\reports\benchmarks\v4_main_culturemert_stage3_lambdamart\eval\dcas_full_ot_calibrated_target.json`
- paired samples: `2400`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| serendipity | 0.5613998658 | 0.8315641066 | +0.2701642408 | [0.264378, 0.275990] | 0.000999 | 1.815469 |
| cultural_calibration_kl | 2.0957527834 | 2.0296378083 | -0.0661149751 | [-0.071767, -0.060896] | 0.000999 | -0.500857 |
| minority_exposure_at_k | 0.2782916667 | 0.4023333333 | +0.1240416667 | [0.116852, 0.130960] | 0.000999 | 0.713543 |

