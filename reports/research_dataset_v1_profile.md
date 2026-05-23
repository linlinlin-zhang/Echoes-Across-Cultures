# Dataset Profile Report

- status: `pass`
- tracks: `1600`
- embedding_dim: `768`
- cultures: `4`
- interactions: `5364`
- users: `120`

## Tracks

- duplicate_track_ids: `0`
- finite_embedding_ratio: `1.0`
- zero_norm_ratio: `0.0`
- culture_imbalance_ratio: `1.0`

### Culture Distribution

| culture | count | ratio |
|---|---:|---:|
| china | 400 | 0.25 |
| india | 400 | 0.25 |
| turkey | 400 | 0.25 |
| west | 400 | 0.25 |

## Interactions

- unknown_track_ratio: `0.0`
- duplicate_user_track_ratio: `0.0`
- non_positive_weight_ratio: `0.0`
- track_coverage_ratio: `0.91875`

## Issues

| severity | code | message |
|---|---|---|
| info | tracks.affect_missing | affect_label is absent (allowed, but limits affect-related evaluation) |
