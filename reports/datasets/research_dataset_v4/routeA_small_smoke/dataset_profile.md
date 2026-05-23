# Dataset Profile

- dataset_name: `research_dataset_v4::routeA_small_smoke`
- metadata_rows: `640`
- cultures: `4`
- sources: `4`

## Culture Distribution

| culture | count | ratio |
|---|---:|---:|
| china | 160 | 0.25 |
| india | 160 | 0.25 |
| turkey | 160 | 0.25 |
| west | 160 | 0.25 |

## Source Distribution

| source_dataset | count | ratio |
|---|---:|---:|
| bilal63/turkish_music_emotion_dataset | 160 | 0.25 |
| ccmusic-database/erhu_playing_tech | 160 | 0.25 |
| neerajaabhyankar/hindustani-raag-small | 160 | 0.25 |
| sanchit-gandhi/gtzan | 160 | 0.25 |

## Source Confound

- single_source_culture_count: `4`
- weighted_source_predictability_from_culture: `1.0`
- weighted_culture_predictability_from_source: `1.0`

| culture | top_source_dataset | top_source_share | n_sources | source_entropy_norm |
|---|---|---:|---:|---:|
| china | ccmusic-database/erhu_playing_tech | 1.0 | 1 | 0.0 |
| india | neerajaabhyankar/hindustani-raag-small | 1.0 | 1 | 0.0 |
| turkey | bilal63/turkish_music_emotion_dataset | 1.0 | 1 | 0.0 |
| west | sanchit-gandhi/gtzan | 1.0 | 1 | 0.0 |

## Interactions

### `interactions_synth_single.csv`

- rows: `3840`
- users: `96`
- track_coverage_ratio: `0.998437`
- unknown_track_ratio: `0.0`
- duplicate_user_track_ratio: `0.0`

### `interactions_synth_mixed.csv`

- rows: `3840`
- users: `96`
- track_coverage_ratio: `0.995313`
- unknown_track_ratio: `0.0`
- duplicate_user_track_ratio: `0.0`

## Issues

| severity | code | message |
|---|---|---|
| warn | metadata.single_source_culture | 4 cultures are backed by a single source dataset |
| warn | metadata.source_confound_high | culture-to-source predictability is high (1.0) |
| warn | metadata.required_field_incomplete | required field 'duration_sec' coverage is only 0.0 |
| warn | metadata.required_field_incomplete | required field 'sample_rate' coverage is only 0.0 |
| warn | metadata.required_field_incomplete | required field 'channels' coverage is only 0.0 |
| warn | metadata.required_field_incomplete | required field 'era' coverage is only 0.0 |
