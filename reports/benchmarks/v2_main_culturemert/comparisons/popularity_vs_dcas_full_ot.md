# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `E:\Desktop\Echo\reports\benchmarks\v2_main_culturemert\eval\popularity.json`
- candidate: `E:\Desktop\Echo\reports\benchmarks\v2_main_culturemert\eval\dcas_full_ot.json`
- paired samples: `600`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| serendipity | 0.6006202852 | 0.8378055970 | +0.2371853118 | [0.219453, 0.253077] | 0.003322 | 1.142736 |
| cultural_calibration_kl | 0.8848024149 | 0.8052499554 | -0.0795524595 | [-0.090451, -0.066868] | 0.003322 | -0.533824 |
| minority_exposure_at_k | 0.0000000000 | 0.3400833333 | +0.3400833333 | [0.310333, 0.369937] | 0.003322 | 0.903255 |

