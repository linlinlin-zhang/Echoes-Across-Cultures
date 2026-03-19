# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `E:\Desktop\Echo\reports\benchmarks\v3_main_culturemert_stage3_bprhybrid\eval\bpr_two_stage_hybrid.json`
- candidate: `E:\Desktop\Echo\reports\benchmarks\v3_main_culturemert_stage3_bprhybrid\eval\dcas_full_ot.json`
- paired samples: `2400`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| serendipity | 0.5067411024 | 0.8452264257 | +0.3384853233 | [0.333157, 0.343360] | 0.004975 | 2.418883 |
| cultural_calibration_kl | 2.0586283729 | 2.0429787993 | -0.0156495736 | [-0.022685, -0.009715] | 0.004975 | -0.094663 |
| minority_exposure_at_k | 0.2023125000 | 0.2397916667 | +0.0374791667 | [0.030561, 0.043755] | 0.004975 | 0.233633 |

