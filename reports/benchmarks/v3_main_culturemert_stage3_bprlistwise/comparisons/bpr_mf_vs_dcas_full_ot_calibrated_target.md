# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `E:\Desktop\Echo\reports\benchmarks\v3_main_culturemert_stage3_bprlistwise\eval\bpr_mf.json`
- candidate: `E:\Desktop\Echo\reports\benchmarks\v3_main_culturemert_stage3_bprlistwise\eval\dcas_full_ot_calibrated_target.json`
- paired samples: `2400`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| serendipity | 0.4915912126 | 0.8385874017 | +0.3469961890 | [0.341878, 0.352252] | 0.004975 | 2.437058 |
| cultural_calibration_kl | 2.0225767313 | 1.8792954780 | -0.1432812533 | [-0.153569, -0.134473] | 0.004975 | -0.608290 |
| minority_exposure_at_k | 0.1491458333 | 0.3814375000 | +0.2322916667 | [0.223871, 0.240483] | 0.004975 | 1.167512 |

