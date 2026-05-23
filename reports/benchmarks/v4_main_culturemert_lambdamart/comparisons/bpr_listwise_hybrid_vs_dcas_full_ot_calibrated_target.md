# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `E:\Desktop\Echo\reports\benchmarks\v4_main_culturemert_lambdamart\eval\bpr_listwise_hybrid.json`
- candidate: `E:\Desktop\Echo\reports\benchmarks\v4_main_culturemert_lambdamart\eval\dcas_full_ot_calibrated_target.json`
- paired samples: `2400`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| serendipity | 0.5671048575 | 0.8110557968 | +0.2439509393 | [0.237841, 0.249728] | 0.000999 | 1.546197 |
| cultural_calibration_kl | 2.0963945231 | 2.0492074270 | -0.0471870961 | [-0.052839, -0.042372] | 0.000999 | -0.372109 |
| minority_exposure_at_k | 0.2775000000 | 0.3998125000 | +0.1223125000 | [0.116186, 0.128689] | 0.000999 | 0.737459 |

