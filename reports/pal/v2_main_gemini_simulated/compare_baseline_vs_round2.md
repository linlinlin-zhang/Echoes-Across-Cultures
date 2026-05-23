# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `E:\Desktop\Echo\reports\pal\v2_main_gemini_simulated\eval_round0_baseline.json`
- candidate: `E:\Desktop\Echo\reports\pal\v2_main_gemini_simulated\eval_round2.json`
- paired samples: `600`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| serendipity | 0.8324788929 | 0.8302076974 | -0.0022711955 | [-0.008159, 0.003025] | 0.458472 | -0.032041 |
| cultural_calibration_kl | 1.9061600792 | 1.9057350357 | -0.0004250435 | [-0.000442, -0.000406] | 0.003322 | -1.811725 |
| minority_exposure_at_k | 0.3615000000 | 0.3705000000 | +0.0090000000 | [0.003246, 0.014750] | 0.013289 | 0.120411 |

