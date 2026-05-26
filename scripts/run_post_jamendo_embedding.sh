#!/usr/bin/env bash
set -euo pipefail

ROOT="${PROJECT_ROOT:-/mnt/e/Desktop/Echo}"
JAMENDO_PID="${JAMENDO_PID:-}"
MODEL_ID="${MODEL_ID:-ntua-slp/CultureMERT-95M}"
MAX_SECONDS="${MAX_SECONDS:-30}"
POOLING="${POOLING:-mean}"

cd "$ROOT"

LOG_DIR="storage/public/merged/logs"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
CHAIN_LOG="$LOG_DIR/jamendo_to_embedding_${STAMP}.log"
EMBED_LOG="$LOG_DIR/culturemert_embedding_${STAMP}.log"

echo "$CHAIN_LOG" > storage/public/merged/jamendo_to_embedding_active_log.txt
echo "$EMBED_LOG" > storage/public/merged/culturemert_embedding_active_log.txt

exec > >(tee -a "$CHAIN_LOG") 2>&1

echo "[START] $(date '+%Y-%m-%d %H:%M:%S')"
echo "[ROOT] $ROOT"

if [[ -n "$JAMENDO_PID" ]]; then
  echo "[WAIT] Waiting for Jamendo PID $JAMENDO_PID"
  while kill -0 "$JAMENDO_PID" 2>/dev/null; do
    sleep 60
  done
  echo "[WAIT] Jamendo PID $JAMENDO_PID has exited"
else
  echo "[WAIT] No JAMENDO_PID supplied; continuing immediately"
fi

echo "[MERGE] Building final merged metadata"
python3 -m dcas.scripts.merge_metadata_dedup \
  --inputs storage/public/itunes_crawl/metadata.csv storage/public/jamendo_crawl/metadata.csv \
  --out storage/public/merged/metadata_merged.csv \
  --require_audio_exists

echo "[ENRICH] Adding cover art and platform playback links"
python3 -m dcas.scripts.enrich_metadata_media_links \
  --metadata storage/public/merged/metadata_merged.csv

echo "[EMBED] Starting CultureMERT full embedding"
echo "[EMBED] Log: $EMBED_LOG"
env PYTHONUNBUFFERED=1 .venv-gpu/Scripts/python.exe -m dcas.scripts.build_tracks_from_audio \
  --metadata storage/public/merged/metadata_merged.csv \
  --out storage/public/merged/tracks_culturemert.npz \
  --model_id "$MODEL_ID" \
  --pooling "$POOLING" \
  --max_seconds "$MAX_SECONDS" \
  --skip_errors 2>&1 | tee -a "$EMBED_LOG"

echo "[DONE] $(date '+%Y-%m-%d %H:%M:%S')"
