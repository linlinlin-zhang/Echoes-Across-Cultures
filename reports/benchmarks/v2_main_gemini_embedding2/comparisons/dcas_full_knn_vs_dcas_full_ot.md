# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `E:\Desktop\Echo\reports\benchmarks\v2_main_gemini_embedding2\eval\dcas_full_knn.json`
- candidate: `E:\Desktop\Echo\reports\benchmarks\v2_main_gemini_embedding2\eval\dcas_full_ot.json`
- paired samples: `600`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| serendipity | 0.8315486778 | 0.8324788929 | +0.0009302151 | [-0.000501, 0.002431] | 0.186047 | 0.048660 |
| cultural_calibration_kl | 1.7592273905 | 1.7591988084 | -0.0000285821 | [-0.000369, 0.000314] | 0.893688 | -0.006343 |
| minority_exposure_at_k | 0.3610000000 | 0.3615000000 | +0.0005000000 | [-0.001377, 0.002417] | 0.518272 | 0.021809 |

