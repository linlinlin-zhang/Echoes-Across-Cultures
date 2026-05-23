# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `E:\Desktop\Echo\reports\benchmarks\v3_main_culturemert_stage3_dcascal\eval\bpr_mf.json`
- candidate: `E:\Desktop\Echo\reports\benchmarks\v3_main_culturemert_stage3_dcascal\eval\dcas_full_ot.json`
- paired samples: `2400`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| serendipity | 0.4915912126 | 0.8452264257 | +0.3536352131 | [0.348263, 0.358988] | 0.004975 | 2.467492 |
| cultural_calibration_kl | 2.0225767313 | 2.0429787993 | +0.0204020679 | [0.012394, 0.027992] | 0.004975 | 0.108260 |
| minority_exposure_at_k | 0.1491458333 | 0.2397916667 | +0.0906458333 | [0.083807, 0.097085] | 0.004975 | 0.545294 |

