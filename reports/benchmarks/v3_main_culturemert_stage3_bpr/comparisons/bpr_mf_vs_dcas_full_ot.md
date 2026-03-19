# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `E:\Desktop\Echo\reports\benchmarks\v3_main_culturemert_stage3_bpr\eval\bpr_mf.json`
- candidate: `E:\Desktop\Echo\reports\benchmarks\v3_main_culturemert_stage3_bpr\eval\dcas_full_ot.json`
- paired samples: `2400`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| serendipity | 0.4915912126 | 0.8448537518 | +0.3532625392 | [0.347996, 0.358553] | 0.003322 | 2.463777 |
| cultural_calibration_kl | 2.0225767313 | 2.0432477311 | +0.0206709997 | [0.013061, 0.027303] | 0.003322 | 0.109877 |
| minority_exposure_at_k | 0.1491458333 | 0.2392083333 | +0.0900625000 | [0.083385, 0.096358] | 0.003322 | 0.540024 |

