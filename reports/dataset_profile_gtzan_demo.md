# Dataset Profile Report

- status: `warn`
- tracks: `12`
- embedding_dim: `768`
- cultures: `1`
- interactions: `20`
- users: `4`

## Tracks

- duplicate_track_ids: `0`
- finite_embedding_ratio: `1.0`
- zero_norm_ratio: `0.0`
- culture_imbalance_ratio: `1.0`

### Culture Distribution

| culture | count | ratio |
|---|---:|---:|
| west | 12 | 1.0 |

## Interactions

- unknown_track_ratio: `0.0`
- duplicate_user_track_ratio: `0.0`
- non_positive_weight_ratio: `0.0`
- track_coverage_ratio: `1.0`

## Issues

| severity | code | message |
|---|---|---|
| warn | tracks.culture_low_count | culture 'west' has only 12 tracks (< 30) |
| info | tracks.affect_missing | affect_label is absent (allowed, but limits affect-related evaluation) |
