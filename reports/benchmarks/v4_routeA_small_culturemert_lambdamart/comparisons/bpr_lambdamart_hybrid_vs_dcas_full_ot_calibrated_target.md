# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `E:\Desktop\Echo\reports\benchmarks\v4_routeA_small_culturemert_lambdamart\eval\bpr_lambdamart_hybrid.json`
- candidate: `E:\Desktop\Echo\reports\benchmarks\v4_routeA_small_culturemert_lambdamart\eval\dcas_full_ot_calibrated_target.json`
- paired samples: `384`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| serendipity | 0.5021775002 | 0.8468967773 | +0.3447192770 | [0.328019, 0.361390] | 0.000999 | 2.066533 |
| cultural_calibration_kl | 1.1068114907 | 1.0872535167 | -0.0195579740 | [-0.025350, -0.013297] | 0.000999 | -0.334353 |
| minority_exposure_at_k | 0.1746093750 | 0.4427083333 | +0.2680989583 | [0.254297, 0.283470] | 0.000999 | 1.874250 |

