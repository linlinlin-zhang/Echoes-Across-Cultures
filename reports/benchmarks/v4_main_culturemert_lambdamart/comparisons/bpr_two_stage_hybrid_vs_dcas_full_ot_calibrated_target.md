# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `E:\Desktop\Echo\reports\benchmarks\v4_main_culturemert_lambdamart\eval\bpr_two_stage_hybrid.json`
- candidate: `E:\Desktop\Echo\reports\benchmarks\v4_main_culturemert_lambdamart\eval\dcas_full_ot_calibrated_target.json`
- paired samples: `2400`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| serendipity | 0.5605093003 | 0.8110557968 | +0.2505464965 | [0.243140, 0.256375] | 0.004975 | 1.585372 |
| cultural_calibration_kl | 2.1005271714 | 2.0492074270 | -0.0513197444 | [-0.056798, -0.046657] | 0.004975 | -0.393579 |
| minority_exposure_at_k | 0.2652708333 | 0.3998125000 | +0.1345416667 | [0.128804, 0.140460] | 0.004975 | 0.833941 |

