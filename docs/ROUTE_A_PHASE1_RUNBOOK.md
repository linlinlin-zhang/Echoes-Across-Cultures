# Route A Phase 1 Runbook

Date: 2026-02-27

This runbook executes Route A with a small multi-cultural public dataset pilot:
- freeze backbone embedding extraction (CultureMERT)
- train DCAS downstream model
- evaluate OT recommendation in batch

## 1) Import three public datasets

```bash
python -m dcas.scripts.import_hf_audio_dataset \
  --dataset sanchit-gandhi/gtzan \
  --split train \
  --streaming \
  --limit 24 \
  --out_dir ./storage/public/routeA_phase1/gtzan \
  --culture_mode constant \
  --culture_value west \
  --label_column genre
```

```bash
python -m dcas.scripts.import_hf_audio_dataset \
  --dataset neerajaabhyankar/hindustani-raag-small \
  --split train \
  --streaming \
  --limit 24 \
  --out_dir ./storage/public/routeA_phase1/hindustani \
  --culture_mode constant \
  --culture_value india \
  --label_column label
```

```bash
python -m dcas.scripts.import_hf_audio_dataset \
  --dataset bilal63/turkish_music_emotion_dataset \
  --split train \
  --streaming \
  --limit 24 \
  --out_dir ./storage/public/routeA_phase1/turkish \
  --culture_mode constant \
  --culture_value turkey \
  --label_column label
```

## 2) Merge metadata and synthesize interactions

```bash
python -m dcas.scripts.merge_metadata \
  --inputs \
    ./storage/public/routeA_phase1/gtzan/metadata.csv \
    ./storage/public/routeA_phase1/hindustani/metadata.csv \
    ./storage/public/routeA_phase1/turkish/metadata.csv \
  --out ./storage/public/routeA_phase1/metadata_merged.csv
```

```bash
python -m dcas.scripts.synthesize_interactions \
  --metadata ./storage/public/routeA_phase1/metadata_merged.csv \
  --out ./storage/public/routeA_phase1/interactions.csv \
  --users_per_culture 8 \
  --tracks_per_user 12 \
  --seed 42
```

## 3) Build embeddings, validate, split

```bash
python -m dcas.scripts.build_tracks_from_audio \
  --metadata ./storage/public/routeA_phase1/metadata_merged.csv \
  --out ./storage/public/routeA_phase1/tracks.npz \
  --model_id ntua-slp/CultureMERT-95M \
  --pooling mean \
  --max_seconds 6 \
  --skip_errors
```

```bash
python -m dcas.scripts.validate_dataset \
  --tracks ./storage/public/routeA_phase1/tracks.npz \
  --interactions ./storage/public/routeA_phase1/interactions.csv \
  --out_json ./reports/routeA_phase1_dataset_profile.json \
  --out_md ./reports/routeA_phase1_dataset_profile.md
```

```bash
python -m dcas.scripts.make_splits \
  --tracks ./storage/public/routeA_phase1/tracks.npz \
  --interactions ./storage/public/routeA_phase1/interactions.csv \
  --out_dir ./reports/routeA_phase1_splits \
  --seed 42 \
  --train_ratio 0.8 \
  --val_ratio 0.1 \
  --test_ratio 0.1
```

## 4) Train and evaluate

```bash
python -m dcas.cli.train \
  --data ./storage/public/routeA_phase1/tracks.npz \
  --out ./storage/public/routeA_phase1/model.pt \
  --epochs 8 \
  --batch_size 32 \
  --lr 2e-3
```

```bash
python -m dcas.scripts.evaluate_recommender \
  --model ./storage/public/routeA_phase1/model.pt \
  --tracks ./storage/public/routeA_phase1/tracks.npz \
  --interactions ./storage/public/routeA_phase1/interactions.csv \
  --out_json ./reports/routeA_phase1_eval.json \
  --k 10
```
