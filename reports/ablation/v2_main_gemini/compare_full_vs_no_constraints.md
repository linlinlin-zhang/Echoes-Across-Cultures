# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `E:\Desktop\Echo\reports\ablation\v2_main_gemini\ablation_full.json`
- candidate: `E:\Desktop\Echo\reports\ablation\v2_main_gemini\ablation_no_constraints.json`
- paired samples: `600`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| serendipity | 0.8224808805 | 0.8413235804 | +0.0188426999 | [0.013505, 0.023517] | 0.003322 | 0.291059 |
| cultural_calibration_kl | 1.9060458933 | 1.9060395192 | -0.0000063741 | [-0.000017, 0.000004] | 0.242525 | -0.047550 |
| minority_exposure_at_k | 0.3629166667 | 0.3385833333 | -0.0243333333 | [-0.030754, -0.017163] | 0.003322 | -0.281064 |

