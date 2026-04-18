# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `E:\Desktop\Echo\reports\pal\v4_main_culturemert_real_from_v4_main_annotation_stage3_ultralight_p3\baseline_eval.json`
- candidate: `E:\Desktop\Echo\reports\pal\v4_main_culturemert_real_from_v4_main_annotation_stage3_ultralight_p3\real_pal_eval.json`
- paired samples: `2400`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| serendipity | 0.8322599439 | 0.8366039213 | +0.0043439775 | [0.002497, 0.006374] | 0.003322 | 0.088869 |
| cultural_calibration_kl | 2.3759675927 | 2.3760844632 | +0.0001168705 | [0.000114, 0.000119] | 0.003322 | 1.872932 |
| minority_exposure_at_k | 0.4499791667 | 0.4668125000 | +0.0168333333 | [0.013206, 0.020189] | 0.003322 | 0.192289 |
| target_culture_prob_mean | 0.1000462573 | 0.1000235229 | -0.0000227344 | [-0.000023, -0.000022] | 0.003322 | -1.872865 |

