# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `E:\Desktop\Echo\reports\benchmarks\v2_main_culturemert\eval\hybrid_content_popularity_diversity.json`
- candidate: `E:\Desktop\Echo\reports\benchmarks\v2_main_culturemert\eval\dcas_full_ot.json`
- paired samples: `600`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| serendipity | 0.6600053128 | 0.8378055970 | +0.1778002842 | [0.163074, 0.191872] | 0.003322 | 0.949737 |
| cultural_calibration_kl | 1.1117159037 | 0.8052499554 | -0.3064659483 | [-0.334325, -0.280839] | 0.003322 | -0.939495 |
| minority_exposure_at_k | 0.1716666667 | 0.3400833333 | +0.1684166667 | [0.152408, 0.185508] | 0.003322 | 0.833961 |

