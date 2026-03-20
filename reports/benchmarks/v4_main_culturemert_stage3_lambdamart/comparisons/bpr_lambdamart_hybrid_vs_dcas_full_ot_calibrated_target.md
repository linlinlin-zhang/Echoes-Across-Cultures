# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `E:\Desktop\Echo\reports\benchmarks\v4_main_culturemert_stage3_lambdamart\eval\bpr_lambdamart_hybrid.json`
- candidate: `E:\Desktop\Echo\reports\benchmarks\v4_main_culturemert_stage3_lambdamart\eval\dcas_full_ot_calibrated_target.json`
- paired samples: `2400`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| serendipity | 0.5558271028 | 0.8315641066 | +0.2757370038 | [0.268045, 0.280856] | 0.004975 | 1.853773 |
| cultural_calibration_kl | 2.0965806578 | 2.0296378083 | -0.0669428495 | [-0.071698, -0.061019] | 0.004975 | -0.504893 |
| minority_exposure_at_k | 0.2680833333 | 0.4023333333 | +0.1342500000 | [0.127473, 0.140370] | 0.004975 | 0.759504 |

