# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `E:\Desktop\Echo\reports\benchmarks\v3_main_culturemert_stage3_stronghybrid\eval\shallow_mlp.json`
- candidate: `E:\Desktop\Echo\reports\benchmarks\v3_main_culturemert_stage3_stronghybrid\eval\dcas_full_ot.json`
- paired samples: `2400`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| serendipity | 0.4812983159 | 0.8448537518 | +0.3635554360 | [0.356271, 0.370082] | 0.003322 | 2.086152 |
| cultural_calibration_kl | 2.1309744927 | 2.0432477311 | -0.0877267616 | [-0.094589, -0.080484] | 0.003322 | -0.497746 |
| minority_exposure_at_k | 0.1673958333 | 0.2392083333 | +0.0718125000 | [0.067246, 0.076283] | 0.003322 | 0.607427 |

