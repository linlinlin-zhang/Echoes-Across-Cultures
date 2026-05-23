# Baseline Comparison Draft Table

| variant | serendipity_mean+/-std | calibration_kl_mean+/-std | minority@k_mean+/-std | delta_ser (full-baseline) | delta_ckl (full-baseline) | delta_minority (full-baseline) |
|---|---:|---:|---:|---:|---:|---:|
| three_factor_dcas | 0.831674 +/- 0.006574 | 2.376078 +/- 0.000010 | 0.424250 +/- 0.010398 | - | - | - |
| vae | 0.854679 +/- 0.001108 | 2.376154 +/- 0.000007 | 0.406111 +/- 0.001292 | -0.023005 | -0.000076 | +0.018139 |
| beta_vae | 0.854026 +/- 0.001494 | 2.376156 +/- 0.000009 | 0.406375 +/- 0.001283 | -0.022352 | -0.000078 | +0.017875 |
| factorvae | 0.846564 +/- 0.007683 | 2.376187 +/- 0.000003 | 0.423208 +/- 0.007742 | -0.014890 | -0.000108 | +0.001042 |

## Necessity Checks

- all_baselines_support_three_factor: `False`
- vae: `False`
- beta_vae: `False`
- factorvae: `False`
