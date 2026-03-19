# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `E:\Desktop\Echo\reports\benchmarks\v3_main_culturemert_stage3\eval\knn.json`
- candidate: `E:\Desktop\Echo\reports\benchmarks\v3_main_culturemert_stage3\eval\dcas_full_ot.json`
- paired samples: `2400`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| serendipity | 0.5704624771 | 0.8448537518 | +0.2743912747 | [0.266152, 0.281573] | 0.003322 | 1.403981 |
| cultural_calibration_kl | 2.1846338018 | 2.0432477311 | -0.1413860707 | [-0.149160, -0.134137] | 0.003322 | -0.739364 |
| minority_exposure_at_k | 0.2263541667 | 0.2392083333 | +0.0128541667 | [0.009333, 0.016734] | 0.003322 | 0.132742 |

