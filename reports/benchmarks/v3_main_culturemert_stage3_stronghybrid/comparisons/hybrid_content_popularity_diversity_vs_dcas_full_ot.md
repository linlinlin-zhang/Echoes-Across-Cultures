# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `E:\Desktop\Echo\reports\benchmarks\v3_main_culturemert_stage3_stronghybrid\eval\hybrid_content_popularity_diversity.json`
- candidate: `E:\Desktop\Echo\reports\benchmarks\v3_main_culturemert_stage3_stronghybrid\eval\dcas_full_ot.json`
- paired samples: `2400`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| serendipity | 0.5368679993 | 0.8448537518 | +0.3079857526 | [0.300100, 0.314894] | 0.003322 | 1.664055 |
| cultural_calibration_kl | 2.1638243315 | 2.0432477311 | -0.1205766004 | [-0.128230, -0.114535] | 0.003322 | -0.696963 |
| minority_exposure_at_k | 0.0700208333 | 0.2392083333 | +0.1691875000 | [0.165531, 0.173462] | 0.003322 | 1.587045 |

