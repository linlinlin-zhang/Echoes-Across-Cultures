# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `E:\Desktop\Echo\reports\benchmarks\v3_main_culturemert_stage3_stronghybrid\eval\two_stage_hybrid_ranker.json`
- candidate: `E:\Desktop\Echo\reports\benchmarks\v3_main_culturemert_stage3_stronghybrid\eval\dcas_full_ot.json`
- paired samples: `2400`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| serendipity | 0.5254739653 | 0.8448537518 | +0.3193797865 | [0.312982, 0.324836] | 0.003322 | 2.043724 |
| cultural_calibration_kl | 2.1521615019 | 2.0432477311 | -0.1089137708 | [-0.116003, -0.103145] | 0.003322 | -0.608971 |
| minority_exposure_at_k | 0.0221458333 | 0.2392083333 | +0.2170625000 | [0.211553, 0.223083] | 0.003322 | 1.457157 |

