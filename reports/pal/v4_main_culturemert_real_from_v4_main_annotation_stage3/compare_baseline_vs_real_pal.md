# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `E:\Desktop\Echo\reports\pal\v4_main_culturemert_real_from_v4_main_annotation_stage3\baseline_eval.json`
- candidate: `E:\Desktop\Echo\reports\pal\v4_main_culturemert_real_from_v4_main_annotation_stage3\real_pal_eval.json`
- paired samples: `2400`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| serendipity | 0.8315708473 | 0.8369521178 | +0.0053812705 | [0.003092, 0.007938] | 0.003322 | 0.086235 |
| cultural_calibration_kl | 2.3759636439 | 2.3760648673 | +0.0001012234 | [0.000099, 0.000103] | 0.003322 | 2.054635 |
| minority_exposure_at_k | 0.4023541667 | 0.3865625000 | -0.0157916667 | [-0.020485, -0.010622] | 0.003322 | -0.119347 |
| target_culture_prob_mean | 0.1000470255 | 0.1000273346 | -0.0000196909 | [-0.000020, -0.000019] | 0.003322 | -2.054598 |

