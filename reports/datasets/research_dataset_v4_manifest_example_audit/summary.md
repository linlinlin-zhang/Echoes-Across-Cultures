# Dataset Manifest Audit

- manifest: `configs/dataset/research_dataset_v4_manifest.example.json`
- dataset_name: `research_dataset_v4`
- dataset_version: `v4_example`
- schema_version: `v4.0`
- sources: `2`

## Cultures

| culture | manifest_source_count |
|---|---:|
| china | 1 |
| india | 1 |

## Source Audit

| dataset_id | culture | exists | rows | duplicate_track_ids | missing_audio_rows |
|---|---|---|---:|---:|---:|
| example/source_a | china | true | 145 | 0 | 0 |
| example/source_b | india | true | 108 | 0 | 0 |

## Issues

| severity | code | message |
|---|---|---|
| info | manifest.single_source_cultures | cultures backed by a single manifest source: china, india |
