#!/usr/bin/env bash
set -euo pipefail

ROOT="${PROJECT_ROOT:-/mnt/e/Desktop/Echo}"
OUT_DIR="${OUT_DIR:-./storage/public/itunes_crawl}"
COUNT_METADATA="${COUNT_METADATA:-./storage/public/jamendo_crawl/metadata.csv}"
MERGE_OUT="${MERGE_OUT:-./storage/public/merged/metadata_merged.csv}"
MODEL_ID="${MODEL_ID:-ntua-slp/CultureMERT-95M}"
MAX_SECONDS="${MAX_SECONDS:-30}"
POOLING="${POOLING:-mean}"
WORKERS="${WORKERS:-4}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-300}"
MAX_PER_QUERY="${MAX_PER_QUERY:-50}"
BATCH_SIZE="${BATCH_SIZE:-120}"
RESTART_DELAY="${RESTART_DELAY:-120}"
IDLE_ROUND_LIMIT="${IDLE_ROUND_LIMIT:-8}"
TARGETS="${TARGETS:-china=1200,japan=1200,korea=1200,india=1200,brazil=1200,latin=1200,africa=1200,middle_east=1200,southeast_asia=1200,celtic=1200,nordic=1200,eastern_europe=1200,balkans=1200,caribbean=1200,andean=1199,central_asia=1199}"

cd "$ROOT"

LOG_DIR="storage/public/merged/logs"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
CHAIN_LOG="$LOG_DIR/nonwest_30k_to_embedding_${STAMP}.log"
EMBED_LOG="$LOG_DIR/culturemert_embedding_${STAMP}.log"

echo "$CHAIN_LOG" > storage/public/merged/nonwest_30k_to_embedding_active_log.txt
echo "$EMBED_LOG" > storage/public/merged/culturemert_embedding_active_log.txt

exec > >(tee -a "$CHAIN_LOG") 2>&1

echo "[START] $(date '+%Y-%m-%d %H:%M:%S')"
echo "[ROOT] $ROOT"
echo "[TARGETS] $TARGETS"

echo "[ITUNES] Supervising non-western top-up toward 30000 total tracks"
PYTHONUNBUFFERED=1 python3 -m dcas.scripts.supervise_itunes_culture_crawl \
  --out_dir "$OUT_DIR" \
  --targets "$TARGETS" \
  --workers "$WORKERS" \
  --checkpoint_interval "$CHECKPOINT_INTERVAL" \
  --max_per_query "$MAX_PER_QUERY" \
  --batch_size "$BATCH_SIZE" \
  --restart_delay "$RESTART_DELAY" \
  --idle_round_limit "$IDLE_ROUND_LIMIT" \
  --count_metadata "$COUNT_METADATA" \
  --merge_out "$MERGE_OUT"

echo "[ENRICH] Adding cover art and platform playback links"
python3 -m dcas.scripts.enrich_metadata_media_links \
  --metadata "$MERGE_OUT"

echo "[EMBED] Starting CultureMERT full embedding"
echo "[EMBED] Log: $EMBED_LOG"
env PYTHONUNBUFFERED=1 .venv-gpu/Scripts/python.exe -m dcas.scripts.build_tracks_from_audio \
  --metadata "$MERGE_OUT" \
  --out storage/public/merged/tracks_culturemert.npz \
  --model_id "$MODEL_ID" \
  --pooling "$POOLING" \
  --max_seconds "$MAX_SECONDS" \
  --skip_errors 2>&1 | tee -a "$EMBED_LOG"

echo "[DONE] $(date '+%Y-%m-%d %H:%M:%S')"
