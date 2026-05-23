# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `E:\Desktop\Echo\reports\benchmarks\v3_main_culturemert_stage3_bpr\eval\dcas_full_knn.json`
- candidate: `E:\Desktop\Echo\reports\benchmarks\v3_main_culturemert_stage3_bpr\eval\dcas_full_ot.json`
- paired samples: `2400`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| serendipity | 0.8454375306 | 0.8448537518 | -0.0005837788 | [-0.001237, 0.000039] | 0.089701 | -0.035990 |
| cultural_calibration_kl | 2.0416590158 | 2.0432477311 | +0.0015887153 | [0.000612, 0.002349] | 0.003322 | 0.071169 |
| minority_exposure_at_k | 0.2395000000 | 0.2392083333 | -0.0002916667 | [-0.001292, 0.000729] | 0.594684 | -0.010831 |

