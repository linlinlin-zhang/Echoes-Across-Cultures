# Dataset Profile Report

- status: `warn`
- tracks: `640`
- embedding_dim: `768`
- cultures: `4`
- interactions: `2682`
- users: `80`

## Tracks

- duplicate_track_ids: `0`
- finite_embedding_ratio: `1.0`
- zero_norm_ratio: `0.0`
- culture_imbalance_ratio: `1.0`

### Culture Distribution

| culture | count | ratio |
|---|---:|---:|
| china | 160 | 0.25 |
| india | 160 | 0.25 |
| turkey | 160 | 0.25 |
| west | 160 | 0.25 |

## Interactions

- unknown_track_ratio: `0.0`
- duplicate_user_track_ratio: `0.0`
- non_positive_weight_ratio: `0.0`
- track_coverage_ratio: `1.0`

## Issues

| severity | code | message |
|---|---|---|
| info | tracks.affect_missing | affect_label is absent (allowed, but limits affect-related evaluation) |
| warn | interactions.low_user_activity | some users have fewer than 5 interactions |
