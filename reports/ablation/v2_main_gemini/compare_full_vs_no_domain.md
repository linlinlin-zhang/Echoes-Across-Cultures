# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `E:\Desktop\Echo\reports\ablation\v2_main_gemini\ablation_full.json`
- candidate: `E:\Desktop\Echo\reports\ablation\v2_main_gemini\ablation_no_domain.json`
- paired samples: `600`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| serendipity | 0.8224808805 | 0.8311560610 | +0.0086751805 | [0.003697, 0.013483] | 0.003322 | 0.140474 |
| cultural_calibration_kl | 1.9060458933 | 1.9057577249 | -0.0002881684 | [-0.000303, -0.000272] | 0.003322 | -1.529712 |
| minority_exposure_at_k | 0.3629166667 | 0.3471666667 | -0.0157500000 | [-0.022548, -0.008956] | 0.003322 | -0.194479 |

