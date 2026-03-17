# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `E:\Desktop\Echo\reports\benchmarks\v2_main_gemini_embedding2\eval\shallow_mlp.json`
- candidate: `E:\Desktop\Echo\reports\benchmarks\v2_main_gemini_embedding2\eval\dcas_full_ot.json`
- paired samples: `600`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| serendipity | 0.8510722322 | 0.8324788929 | -0.0185933393 | [-0.026131, -0.009571] | 0.003322 | -0.167184 |
| cultural_calibration_kl | 1.8078915778 | 1.7591988084 | -0.0486927694 | [-0.052790, -0.044028] | 0.003322 | -0.885791 |
| minority_exposure_at_k | 0.3540000000 | 0.3615000000 | +0.0075000000 | [0.001571, 0.014298] | 0.043189 | 0.094900 |

