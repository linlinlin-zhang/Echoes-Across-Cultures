# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `E:\Desktop\Echo\reports\benchmarks\v2_main_gemini_embedding2\eval\knn.json`
- candidate: `E:\Desktop\Echo\reports\benchmarks\v2_main_gemini_embedding2\eval\dcas_full_ot.json`
- paired samples: `600`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| serendipity | 0.8978959201 | 0.8324788929 | -0.0654170272 | [-0.072763, -0.057102] | 0.003322 | -0.678268 |
| cultural_calibration_kl | 1.8020711659 | 1.7591988084 | -0.0428723575 | [-0.046681, -0.038183] | 0.003322 | -0.787573 |
| minority_exposure_at_k | 0.3435833333 | 0.3615000000 | +0.0179166667 | [0.012250, 0.024167] | 0.003322 | 0.240121 |

