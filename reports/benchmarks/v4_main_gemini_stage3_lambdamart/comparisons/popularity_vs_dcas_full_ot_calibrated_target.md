# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `E:\Desktop\Echo\reports\benchmarks\v4_main_gemini_stage3_lambdamart\eval\popularity.json`
- candidate: `E:\Desktop\Echo\reports\benchmarks\v4_main_gemini_stage3_lambdamart\eval\dcas_full_ot_calibrated_target.json`
- paired samples: `2400`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| serendipity | 0.7576204119 | 0.8244925563 | +0.0668721444 | [0.062276, 0.071224] | 0.000999 | 0.615170 |
| cultural_calibration_kl | 2.3290537030 | 2.3104056591 | -0.0186480438 | [-0.020250, -0.016946] | 0.000999 | -0.457413 |
| minority_exposure_at_k | 0.0000000000 | 0.3759791667 | +0.3759791667 | [0.369477, 0.382501] | 0.000999 | 2.231745 |

