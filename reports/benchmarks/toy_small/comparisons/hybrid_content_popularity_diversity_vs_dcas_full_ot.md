# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `E:\Desktop\Echo\reports\benchmarks\toy_small\eval\hybrid_content_popularity_diversity.json`
- candidate: `E:\Desktop\Echo\reports\benchmarks\toy_small\eval\dcas_full_ot.json`
- paired samples: `18`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| serendipity | 0.6928064348 | 0.7978846578 | +0.1050782230 | [0.035430, 0.165999] | 0.019900 | 0.701853 |
| cultural_calibration_kl | 0.0430288016 | 0.0426531376 | -0.0003756640 | [-0.002369, 0.001704] | 0.766169 | -0.078546 |
| minority_exposure_at_k | 0.0944444444 | 0.1277777778 | +0.0333333333 | [-0.016667, 0.077778] | 0.253731 | 0.323942 |

