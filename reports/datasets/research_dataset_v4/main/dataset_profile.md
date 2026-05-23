# Dataset Profile

- dataset_name: `research_dataset_v4::main`
- metadata_rows: `1122`
- cultures: `10`
- sources: `8`

## Culture Distribution

| culture | count | ratio |
|---|---:|---:|
| turkey | 150 | 0.13369 |
| china | 145 | 0.129234 |
| modern_english_pop | 120 | 0.106952 |
| india | 108 | 0.096257 |
| france | 105 | 0.093583 |
| germany | 105 | 0.093583 |
| great_britain | 105 | 0.093583 |
| italy | 105 | 0.093583 |
| russia | 105 | 0.093583 |
| indonesia | 74 | 0.065954 |

## Source Distribution

| source_dataset | count | ratio |
|---|---:|---:|
| Free Music Archive | 544 | 0.484848 |
| bilal63/turkish_music_emotion_dataset | 150 | 0.13369 |
| vtsouval/mtg_jamendo_autotagging | 120 | 0.106952 |
| saraga_hindustani | 108 | 0.096257 |
| ccmusic-database/CTIS | 65 | 0.057932 |
| gamelan_music_dataset | 55 | 0.04902 |
| OpenCpop | 50 | 0.044563 |
| compmusic_jingju_acappella | 30 | 0.026738 |

## Source Confound

- single_source_culture_count: `8`
- weighted_source_predictability_from_culture: `0.911765`
- weighted_culture_predictability_from_source: `0.608735`

| culture | top_source_dataset | top_source_share | n_sources | source_entropy_norm |
|---|---|---:|---:|---:|
| china | ccmusic-database/CTIS | 0.448276 | 3 | 0.958288 |
| france | Free Music Archive | 1.0 | 1 | 0.0 |
| germany | Free Music Archive | 1.0 | 1 | 0.0 |
| great_britain | Free Music Archive | 1.0 | 1 | 0.0 |
| india | saraga_hindustani | 1.0 | 1 | 0.0 |
| indonesia | gamelan_music_dataset | 0.743243 | 2 | 0.821813 |
| italy | Free Music Archive | 1.0 | 1 | 0.0 |
| modern_english_pop | vtsouval/mtg_jamendo_autotagging | 1.0 | 1 | 0.0 |
| russia | Free Music Archive | 1.0 | 1 | 0.0 |
| turkey | bilal63/turkish_music_emotion_dataset | 1.0 | 1 | 0.0 |

## Interactions

### `interactions_synth_single.csv`

- rows: `9600`
- users: `240`
- track_coverage_ratio: `1.0`
- unknown_track_ratio: `0.0`
- duplicate_user_track_ratio: `0.0`

### `interactions_synth_mixed.csv`

- rows: `9600`
- users: `240`
- track_coverage_ratio: `1.0`
- unknown_track_ratio: `0.0`
- duplicate_user_track_ratio: `0.0`

## Issues

| severity | code | message |
|---|---|---|
| warn | metadata.single_source_culture | 8 cultures are backed by a single source dataset |
| warn | metadata.source_confound_high | culture-to-source predictability is high (0.911765) |
