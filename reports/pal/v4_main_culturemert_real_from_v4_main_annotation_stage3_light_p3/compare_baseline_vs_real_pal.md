# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `E:\Desktop\Echo\reports\pal\v4_main_culturemert_real_from_v4_main_annotation_stage3_light_p3\baseline_eval.json`
- candidate: `E:\Desktop\Echo\reports\pal\v4_main_culturemert_real_from_v4_main_annotation_stage3_light_p3\real_pal_eval.json`
- paired samples: `2400`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| serendipity | 0.8322599439 | 0.8372839786 | +0.0050240347 | [0.003155, 0.006941] | 0.003322 | 0.101922 |
| cultural_calibration_kl | 2.3759675927 | 2.3760932240 | +0.0001256313 | [0.000123, 0.000128] | 0.003322 | 1.930663 |
| minority_exposure_at_k | 0.4499791667 | 0.4178750000 | -0.0321041667 | [-0.037889, -0.026903] | 0.003322 | -0.222670 |
| target_culture_prob_mean | 0.1000462573 | 0.1000218185 | -0.0000244388 | [-0.000025, -0.000024] | 0.003322 | -1.930647 |

