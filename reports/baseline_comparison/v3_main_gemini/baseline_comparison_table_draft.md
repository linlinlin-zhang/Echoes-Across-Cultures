# Baseline Comparison Draft Table

| variant | serendipity_mean+/-std | calibration_kl_mean+/-std | minority@k_mean+/-std | delta_ser (full-baseline) | delta_ckl (full-baseline) | delta_minority (full-baseline) |
|---|---:|---:|---:|---:|---:|---:|
| three_factor_dcas | 0.822584 +/- 0.010222 | 2.376040 +/- 0.000024 | 0.390681 +/- 0.020501 | - | - | - |
| vae | 0.837985 +/- 0.004071 | 2.376171 +/- 0.000011 | 0.374472 +/- 0.022446 | -0.015401 | -0.000131 | +0.016208 |
| beta_vae | 0.838226 +/- 0.004091 | 2.376171 +/- 0.000011 | 0.374431 +/- 0.021379 | -0.015643 | -0.000131 | +0.016250 |
| factorvae | 0.832523 +/- 0.003761 | 2.376196 +/- 0.000001 | 0.382417 +/- 0.016115 | -0.009939 | -0.000156 | +0.008264 |

## Necessity Checks

- all_baselines_support_three_factor: `False`
- vae: `False`
- beta_vae: `False`
- factorvae: `False`
