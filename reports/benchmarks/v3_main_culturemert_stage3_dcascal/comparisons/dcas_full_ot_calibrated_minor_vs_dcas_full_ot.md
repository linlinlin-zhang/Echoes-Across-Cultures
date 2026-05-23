# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `E:\Desktop\Echo\reports\benchmarks\v3_main_culturemert_stage3_dcascal\eval\dcas_full_ot_calibrated_minor.json`
- candidate: `E:\Desktop\Echo\reports\benchmarks\v3_main_culturemert_stage3_dcascal\eval\dcas_full_ot.json`
- paired samples: `2400`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| serendipity | 0.8404379939 | 0.8452264257 | +0.0047884318 | [0.002797, 0.006524] | 0.004975 | 0.101086 |
| cultural_calibration_kl | 1.9148024109 | 2.0429787993 | +0.1281763884 | [0.120172, 0.135206] | 0.004975 | 0.601950 |
| minority_exposure_at_k | 0.5189583333 | 0.2397916667 | -0.2791666667 | [-0.285318, -0.271770] | 0.004975 | -1.746016 |

