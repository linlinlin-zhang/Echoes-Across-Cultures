# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `E:\Desktop\Echo\reports\benchmarks\v3_main_culturemert_stage3_bprhybrid\eval\cosine.json`
- candidate: `E:\Desktop\Echo\reports\benchmarks\v3_main_culturemert_stage3_bprhybrid\eval\dcas_full_ot.json`
- paired samples: `2400`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| serendipity | 0.5542390163 | 0.8452264257 | +0.2909874094 | [0.283399, 0.298874] | 0.004975 | 1.484301 |
| cultural_calibration_kl | 2.1760311331 | 2.0429787993 | -0.1330523338 | [-0.139822, -0.125661] | 0.004975 | -0.685637 |
| minority_exposure_at_k | 0.2265416667 | 0.2397916667 | +0.0132500000 | [0.009242, 0.017002] | 0.004975 | 0.130021 |

