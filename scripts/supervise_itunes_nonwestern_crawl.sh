#!/usr/bin/env bash
set -u

cd "$(dirname "$0")/.."

OUT_DIR="${OUT_DIR:-./storage/public/itunes_crawl}"
TARGETS="${TARGETS:-china=1200,japan=1200,korea=1200,india=1200,brazil=1200,latin=1200,africa=1200,middle_east=1200,southeast_asia=1200,celtic=1200,nordic=1200,eastern_europe=1200,balkans=1200,caribbean=1200,andean=1199,central_asia=1199}"
COUNT_METADATA="${COUNT_METADATA:-./storage/public/jamendo_crawl/metadata.csv}"
WORKERS="${WORKERS:-4}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-300}"
MAX_PER_QUERY="${MAX_PER_QUERY:-50}"
BATCH_SIZE="${BATCH_SIZE:-120}"
RESTART_DELAY="${RESTART_DELAY:-120}"
IDLE_ROUND_LIMIT="${IDLE_ROUND_LIMIT:-8}"

PYTHONUNBUFFERED=1 python3 -m dcas.scripts.supervise_itunes_culture_crawl \
  --out_dir "$OUT_DIR" \
  --targets "$TARGETS" \
  --workers "$WORKERS" \
  --checkpoint_interval "$CHECKPOINT_INTERVAL" \
  --max_per_query "$MAX_PER_QUERY" \
  --batch_size "$BATCH_SIZE" \
  --restart_delay "$RESTART_DELAY" \
  --idle_round_limit "$IDLE_ROUND_LIMIT" \
  --count_metadata "$COUNT_METADATA"
