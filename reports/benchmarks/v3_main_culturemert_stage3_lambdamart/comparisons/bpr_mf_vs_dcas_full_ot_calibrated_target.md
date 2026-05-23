# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `E:\Desktop\Echo\reports\benchmarks\v3_main_culturemert_stage3_lambdamart\eval\bpr_mf.json`
- candidate: `E:\Desktop\Echo\reports\benchmarks\v3_main_culturemert_stage3_lambdamart\eval\dcas_full_ot_calibrated_target.json`
- paired samples: `2400`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| serendipity | 0.4915912126 | 0.8385876200 | +0.3469964074 | [0.341878, 0.352252] | 0.004975 | 2.437010 |
| cultural_calibration_kl | 2.0225767313 | 1.8792946220 | -0.1432821094 | [-0.153570, -0.134474] | 0.004975 | -0.608294 |
| minority_exposure_at_k | 0.1491458333 | 0.3814375000 | +0.2322916667 | [0.223871, 0.240483] | 0.004975 | 1.167512 |

