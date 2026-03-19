# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `E:\Desktop\Echo\reports\benchmarks\v3_main_culturemert_stage3_dcascal\eval\bpr_two_stage_hybrid.json`
- candidate: `E:\Desktop\Echo\reports\benchmarks\v3_main_culturemert_stage3_dcascal\eval\dcas_full_ot.json`
- paired samples: `2400`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| serendipity | 0.5101928902 | 0.8452264257 | +0.3350335355 | [0.329698, 0.339603] | 0.004975 | 2.378683 |
| cultural_calibration_kl | 2.0081828249 | 2.0429787993 | +0.0347959744 | [0.027705, 0.041193] | 0.004975 | 0.195673 |
| minority_exposure_at_k | 0.2837500000 | 0.2397916667 | -0.0439583333 | [-0.051522, -0.036979] | 0.004975 | -0.252062 |

