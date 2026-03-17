# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `E:\Desktop\Echo\reports\ablation\v2_main_gemini\ablation_full.json`
- candidate: `E:\Desktop\Echo\reports\ablation\v2_main_gemini\ablation_no_ot.json`
- paired samples: `600`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| serendipity | 0.8224808805 | 0.8238579559 | +0.0013770755 | [-0.000198, 0.003103] | 0.122924 | 0.063937 |
| cultural_calibration_kl | 1.9060458933 | 1.9060464369 | +0.0000005436 | [-0.000000, 0.000001] | 0.295681 | 0.044461 |
| minority_exposure_at_k | 0.3629166667 | 0.3611666667 | -0.0017500000 | [-0.003460, -0.000000] | 0.056478 | -0.073382 |

