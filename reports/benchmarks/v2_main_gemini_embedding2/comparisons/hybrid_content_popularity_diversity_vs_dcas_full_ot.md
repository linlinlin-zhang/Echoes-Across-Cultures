# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `E:\Desktop\Echo\reports\benchmarks\v2_main_gemini_embedding2\eval\hybrid_content_popularity_diversity.json`
- candidate: `E:\Desktop\Echo\reports\benchmarks\v2_main_gemini_embedding2\eval\dcas_full_ot.json`
- paired samples: `600`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| serendipity | 0.8574686307 | 0.8324788929 | -0.0249897378 | [-0.031514, -0.016367] | 0.003322 | -0.250415 |
| cultural_calibration_kl | 1.7945896687 | 1.7591988084 | -0.0353908603 | [-0.039030, -0.030770] | 0.003322 | -0.660893 |
| minority_exposure_at_k | 0.1555000000 | 0.3615000000 | +0.2060000000 | [0.187737, 0.226306] | 0.003322 | 0.875028 |

