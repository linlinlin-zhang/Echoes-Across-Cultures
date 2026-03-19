# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `E:\Desktop\Echo\reports\benchmarks\v3_main_culturemert_stage3_bprhybrid\eval\hybrid_content_popularity_diversity.json`
- candidate: `E:\Desktop\Echo\reports\benchmarks\v3_main_culturemert_stage3_bprhybrid\eval\dcas_full_ot.json`
- paired samples: `2400`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| serendipity | 0.5368679993 | 0.8452264257 | +0.3083584265 | [0.301121, 0.315215] | 0.004975 | 1.665048 |
| cultural_calibration_kl | 2.1638243315 | 2.0429787993 | -0.1208455322 | [-0.127041, -0.113831] | 0.004975 | -0.698731 |
| minority_exposure_at_k | 0.0700208333 | 0.2397916667 | +0.1697708333 | [0.166185, 0.174084] | 0.004975 | 1.587996 |

