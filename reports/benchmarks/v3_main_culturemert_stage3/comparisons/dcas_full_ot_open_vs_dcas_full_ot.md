# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `E:\Desktop\Echo\reports\benchmarks\v3_main_culturemert_stage3\eval\dcas_full_ot_open.json`
- candidate: `E:\Desktop\Echo\reports\benchmarks\v3_main_culturemert_stage3\eval\dcas_full_ot.json`
- paired samples: `2400`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| serendipity | 0.2755798067 | 0.8448537518 | +0.5692739452 | [0.566636, 0.571708] | 0.003322 | 9.316772 |
| cultural_calibration_kl | 2.3009621209 | 2.0432477311 | -0.2577143899 | [-0.270976, -0.243650] | 0.003322 | -0.854186 |
| minority_exposure_at_k | 0.3724791667 | 0.2392083333 | -0.1332708333 | [-0.139306, -0.127753] | 0.003322 | -0.865462 |

