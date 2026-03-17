# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `E:\Desktop\Echo\reports\benchmarks\v2_main_gemini_embedding2\eval\cosine.json`
- candidate: `E:\Desktop\Echo\reports\benchmarks\v2_main_gemini_embedding2\eval\dcas_full_ot.json`
- paired samples: `600`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| serendipity | 0.8938577778 | 0.8324788929 | -0.0613788849 | [-0.068994, -0.052459] | 0.003322 | -0.593833 |
| cultural_calibration_kl | 1.8012670770 | 1.7591988084 | -0.0420682686 | [-0.045755, -0.037290] | 0.003322 | -0.768291 |
| minority_exposure_at_k | 0.3463333333 | 0.3615000000 | +0.0151666667 | [0.009500, 0.021254] | 0.003322 | 0.206957 |

