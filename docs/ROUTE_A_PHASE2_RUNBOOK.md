# Route A Phase 2 Runbook

Date: 2026-02-27

Goal:
- expand multi-cultural public data scale
- keep backbone frozen (embedding extraction only)
- retrain DCAS and compare with Phase 1

## 1) Import datasets (160 per culture)

```bash
python -m dcas.scripts.import_hf_audio_dataset \
  --dataset sanchit-gandhi/gtzan \
  --split train \
  --streaming \
  --limit 160 \
  --out_dir ./storage/public/routeA_phase2/gtzan \
  --culture_mode constant \
  --culture_value west \
  --label_column genre
```

```bash
python -m dcas.scripts.import_hf_audio_dataset \
  --dataset neerajaabhyankar/hindustani-raag-small \
  --split train \
  --streaming \
  --limit 160 \
  --out_dir ./storage/public/routeA_phase2/hindustani \
  --culture_mode constant \
  --culture_value india \
  --label_column label
```

```bash
python -m dcas.scripts.import_hf_audio_dataset \
  --dataset bilal63/turkish_music_emotion_dataset \
  --split train \
  --streaming \
  --limit 160 \
  --out_dir ./storage/public/routeA_phase2/turkish \
  --culture_mode constant \
  --culture_value turkey \
  --label_column label
```

## 2) Merge metadata and synthesize interactions

```bash
python -m dcas.scripts.merge_metadata \
  --inputs \
    ./storage/public/routeA_phase2/gtzan/metadata.csv \
    ./storage/public/routeA_phase2/hindustani/metadata.csv \
    ./storage/public/routeA_phase2/turkish/metadata.csv \
  --out ./storage/public/routeA_phase2/metadata_merged.csv
```

```bash
python -m dcas.scripts.synthesize_interactions \
  --metadata ./storage/public/routeA_phase2/metadata_merged.csv \
  --out ./storage/public/routeA_phase2/interactions.csv \
  --users_per_culture 20 \
  --tracks_per_user 50 \
  --seed 42
```

## 3) Build embeddings

```bash
python -m dcas.scripts.build_tracks_from_audio \
  --metadata ./storage/public/routeA_phase2/metadata_merged.csv \
  --out ./storage/public/routeA_phase2/tracks.npz \
  --model_id ntua-slp/CultureMERT-95M \
  --pooling mean \
  --max_seconds 6 \
  --skip_errors
```

## 4) Validate and split

```bash
python -m dcas.scripts.validate_dataset \
  --tracks ./storage/public/routeA_phase2/tracks.npz \
  --interactions ./storage/public/routeA_phase2/interactions.csv \
  --out_json ./reports/routeA_phase2_dataset_profile.json \
  --out_md ./reports/routeA_phase2_dataset_profile.md
```

```bash
python -m dcas.scripts.make_splits \
  --tracks ./storage/public/routeA_phase2/tracks.npz \
  --interactions ./storage/public/routeA_phase2/interactions.csv \
  --out_dir ./reports/routeA_phase2_splits \
  --seed 42 \
  --train_ratio 0.8 \
  --val_ratio 0.1 \
  --test_ratio 0.1
```

## 5) Train and evaluate

```bash
python -m dcas.cli.train \
  --data ./storage/public/routeA_phase2/tracks.npz \
  --out ./storage/public/routeA_phase2/model.pt \
  --epochs 10 \
  --batch_size 64 \
  --lr 2e-3
```

```bash
python -m dcas.scripts.evaluate_recommender \
  --model ./storage/public/routeA_phase2/model.pt \
  --tracks ./storage/public/routeA_phase2/tracks.npz \
  --interactions ./storage/public/routeA_phase2/interactions.csv \
  --out_json ./reports/routeA_phase2_eval.json \
  --k 10
```
