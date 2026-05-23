# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `E:\Desktop\Echo\reports\benchmarks\v3_main_culturemert_stage3_bprhybrid\eval\popularity.json`
- candidate: `E:\Desktop\Echo\reports\benchmarks\v3_main_culturemert_stage3_bprhybrid\eval\dcas_full_ot.json`
- paired samples: `2400`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| serendipity | 0.4543026544 | 0.8452264257 | +0.3909237713 | [0.385014, 0.396690] | 0.004975 | 2.353291 |
| cultural_calibration_kl | 2.0709474615 | 2.0429787993 | -0.0279686622 | [-0.038043, -0.019816] | 0.004975 | -0.121295 |
| minority_exposure_at_k | 0.0000000000 | 0.2397916667 | +0.2397916667 | [0.234932, 0.245437] | 0.004975 | 1.632971 |

