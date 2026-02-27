# Dataset Profile Report

- status: `warn`
- tracks: `100`
- embedding_dim: `64`
- cultures: `3`
- interactions: `75`
- users: `6`

## Tracks

- duplicate_track_ids: `0`
- finite_embedding_ratio: `1.0`
- zero_norm_ratio: `0.0`
- culture_imbalance_ratio: `1.428571`

### Culture Distribution

| culture | count | ratio |
|---|---:|---:|
| west | 40 | 0.4 |
| africa | 32 | 0.32 |
| india | 28 | 0.28 |

## Interactions

- unknown_track_ratio: `0.0`
- duplicate_user_track_ratio: `0.0`
- non_positive_weight_ratio: `0.0`
- track_coverage_ratio: `0.75`

## Issues

| severity | code | message |
|---|---|---|
| warn | tracks.culture_low_count | culture 'india' has only 28 tracks (< 30) |
