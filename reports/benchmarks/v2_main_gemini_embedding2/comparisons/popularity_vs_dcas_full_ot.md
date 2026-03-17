# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `E:\Desktop\Echo\reports\benchmarks\v2_main_gemini_embedding2\eval\popularity.json`
- candidate: `E:\Desktop\Echo\reports\benchmarks\v2_main_gemini_embedding2\eval\dcas_full_ot.json`
- paired samples: `600`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| serendipity | 0.8315062467 | 0.8324788929 | +0.0009726462 | [-0.007478, 0.011755] | 0.833887 | 0.007918 |
| cultural_calibration_kl | 1.7596026528 | 1.7591988084 | -0.0004038445 | [-0.002669, 0.002024] | 0.760797 | -0.013531 |
| minority_exposure_at_k | 0.0000000000 | 0.3615000000 | +0.3615000000 | [0.329035, 0.394362] | 0.003322 | 0.916900 |

