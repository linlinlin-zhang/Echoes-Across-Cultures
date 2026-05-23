# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `E:\Desktop\Echo\reports\benchmarks\v3_main_culturemert_stage3_stronghybrid\eval\cosine.json`
- candidate: `E:\Desktop\Echo\reports\benchmarks\v3_main_culturemert_stage3_stronghybrid\eval\dcas_full_ot.json`
- paired samples: `2400`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| serendipity | 0.5542390163 | 0.8448537518 | +0.2906147355 | [0.282209, 0.298268] | 0.003322 | 1.482634 |
| cultural_calibration_kl | 2.1760311331 | 2.0432477311 | -0.1327834020 | [-0.140904, -0.125153] | 0.003322 | -0.684486 |
| minority_exposure_at_k | 0.2265416667 | 0.2392083333 | +0.0126666667 | [0.008809, 0.016658] | 0.003322 | 0.124020 |

