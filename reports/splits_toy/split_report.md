# Dataset Split Report

- status: `pass`
- n_tracks: `1200`
- seed: `42`
- ratios(train/val/test): `0.8/0.1/0.1`

## Split Sizes

| split | n_tracks | ratio |
|---|---:|---:|
| train | 960 | 0.8 |
| val | 120 | 0.1 |
| test | 120 | 0.1 |

## Leakage Check

- has_leakage: `False`
- overlap_count: `0`

## Culture Distribution Drift

- max_abs_delta_from_global: `0.0025`

## Interaction Coverage

- n_rows: `480`
- unknown_track_ratio: `0.0`

| split | interaction_count | ratio | users |
|---|---:|---:|---:|
| train | 380 | 0.791667 | 6 |
| val | 47 | 0.097917 | 6 |
| test | 53 | 0.110417 | 6 |

## Issues

- none
