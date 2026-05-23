# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `E:\Desktop\Echo\reports\pal\v2_main_gemini_simulated\eval_round0_baseline.json`
- candidate: `E:\Desktop\Echo\reports\pal\v2_main_gemini_simulated\eval_round1.json`
- paired samples: `600`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| serendipity | 0.8324788929 | 0.8125835637 | -0.0198953292 | [-0.024728, -0.014561] | 0.003322 | -0.289019 |
| cultural_calibration_kl | 1.9061600792 | 1.9059174409 | -0.0002426383 | [-0.000257, -0.000226] | 0.003322 | -1.246265 |
| minority_exposure_at_k | 0.3615000000 | 0.3706666667 | +0.0091666667 | [0.001956, 0.015631] | 0.009967 | 0.103988 |

