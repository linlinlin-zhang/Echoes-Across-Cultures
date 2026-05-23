# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `E:\Desktop\Echo\reports\benchmarks\v3_main_culturemert_stage3\eval\popularity.json`
- candidate: `E:\Desktop\Echo\reports\benchmarks\v3_main_culturemert_stage3\eval\dcas_full_ot.json`
- paired samples: `2400`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| serendipity | 0.4543026544 | 0.8448537518 | +0.3905510974 | [0.384288, 0.396965] | 0.003322 | 2.352377 |
| cultural_calibration_kl | 2.0709474615 | 2.0432477311 | -0.0276997304 | [-0.038009, -0.019289] | 0.003322 | -0.120012 |
| minority_exposure_at_k | 0.0000000000 | 0.2392083333 | +0.2392083333 | [0.233730, 0.244882] | 0.003322 | 1.631593 |

