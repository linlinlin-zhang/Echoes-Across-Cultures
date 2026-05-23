# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `E:\Desktop\Echo\reports\benchmarks\v3_main_culturemert_stage3_bprlistwise\eval\bpr_listwise_hybrid.json`
- candidate: `E:\Desktop\Echo\reports\benchmarks\v3_main_culturemert_stage3_bprlistwise\eval\dcas_full_ot_calibrated_target.json`
- paired samples: `2400`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| serendipity | 0.5061616138 | 0.8385874017 | +0.3324257879 | [0.327400, 0.337733] | 0.004975 | 2.334661 |
| cultural_calibration_kl | 2.0102720289 | 1.8792954780 | -0.1309765509 | [-0.141973, -0.121441] | 0.004975 | -0.585439 |
| minority_exposure_at_k | 0.2250833333 | 0.3814375000 | +0.1563541667 | [0.146833, 0.164919] | 0.004975 | 0.809513 |

