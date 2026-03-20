# Recommender Run Comparison (Paired Bootstrap + Permutation Test)

- base: `E:\Desktop\Echo\reports\benchmarks\v4_main_gemini_stage3_lambdamart\eval\bpr_two_stage_hybrid.json`
- candidate: `E:\Desktop\Echo\reports\benchmarks\v4_main_gemini_stage3_lambdamart\eval\dcas_full_ot_calibrated_target.json`
- paired samples: `2400`

| metric | base_mean | candidate_mean | delta_mean | 95% CI (delta) | p_value(two-sided) | cohen_d(paired) |
|---|---:|---:|---:|---:|---:|---:|
| serendipity | 0.7870007950 | 0.8244925563 | +0.0374917614 | [0.033324, 0.040896] | 0.004975 | 0.413298 |
| cultural_calibration_kl | 2.3196384795 | 2.3104056591 | -0.0092328203 | [-0.009984, -0.008546] | 0.004975 | -0.505087 |
| minority_exposure_at_k | 0.3025416667 | 0.3759791667 | +0.0734375000 | [0.066580, 0.080503] | 0.004975 | 0.433684 |

