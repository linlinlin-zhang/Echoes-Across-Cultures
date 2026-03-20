# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `E:\Desktop\Echo\reports\benchmarks\v4_routeA_small_gemini_stage3_lambdamart\eval\bpr_mf.json`
- candidate: `E:\Desktop\Echo\reports\benchmarks\v4_routeA_small_gemini_stage3_lambdamart\eval\dcas_full_ot_calibrated_target.json`
- paired samples: `384`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| serendipity | 0.7103953568 | 0.8641977006 | +0.1538023437 | [0.144120, 0.166493] | 0.004975 | 1.460716 |
| cultural_calibration_kl | 1.5722634472 | 1.5501961971 | -0.0220672500 | [-0.023983, -0.020351] | 0.004975 | -1.262244 |
| minority_exposure_at_k | 0.0958333333 | 0.4997395833 | +0.4039062500 | [0.387340, 0.421113] | 0.004975 | 2.457722 |

