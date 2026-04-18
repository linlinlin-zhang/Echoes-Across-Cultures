# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `E:\Desktop\Echo\reports\pal\v4_main_culturemert_real_from_v4_main_annotation\baseline_eval.json`
- candidate: `E:\Desktop\Echo\reports\pal\v4_main_culturemert_real_from_v4_main_annotation\real_pal_eval.json`
- paired samples: `2400`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| serendipity | 0.8579567105 | 0.8475947987 | -0.0103619118 | [-0.012372, -0.008276] | 0.003322 | -0.191389 |
| cultural_calibration_kl | 2.3760583116 | 2.3760313356 | -0.0000269760 | [-0.000030, -0.000024] | 0.003322 | -0.376623 |
| minority_exposure_at_k | 0.2470416667 | 0.2195416667 | -0.0275000000 | [-0.032131, -0.022744] | 0.003322 | -0.243374 |

