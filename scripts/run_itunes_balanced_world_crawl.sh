#!/usr/bin/env bash
set -u

cd "$(dirname "$0")/.."

OUT_DIR="./storage/public/itunes_crawl"
PER_COUNTRY="${PER_COUNTRY:-200}"
MAX_PER_QUERY="${MAX_PER_QUERY:-50}"
WORKERS="${WORKERS:-4}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-300}"

COUNTRIES="$(
  python3 -c 'from dcas.scripts.crawl_itunes_previews import DEFAULT_COUNTRIES; print(" ".join(DEFAULT_COUNTRIES))'
)"

echo "[BALANCED CRAWL START] $(date -Is)"
echo "out_dir=$OUT_DIR"
echo "per_country=$PER_COUNTRY max_per_query=$MAX_PER_QUERY workers=$WORKERS"
echo "countries=$COUNTRIES"

for country in $COUNTRIES; do
  current="$(
    python3 -c 'import csv, pathlib; p=pathlib.Path("storage/public/itunes_crawl/metadata.csv"); print(len({row.get("track_id", "") for row in csv.DictReader(p.open(encoding="utf-8", newline="")) if row.get("track_id", "")}) if p.exists() else 0)'
  )"
  country_count="$(
    COUNTRY="$country" python3 -c 'import csv, os, pathlib; p=pathlib.Path("storage/public/itunes_crawl/metadata.csv"); country=os.environ["COUNTRY"]; print(len({row.get("track_id", "") for row in csv.DictReader(p.open(encoding="utf-8", newline="")) if row.get("country") == country and row.get("track_id", "")}) if p.exists() else 0)'
  )"
  if [ "$country_count" -ge "$PER_COUNTRY" ]; then
    echo "[COUNTRY SKIP] $(date -Is) country=$country country_count=$country_count target_per_country=$PER_COUNTRY"
    continue
  fi
  needed=$((PER_COUNTRY - country_count))
  target=$((current + needed))
  echo "[COUNTRY START] $(date -Is) country=$country current=$current country_count=$country_count target=$target"
  python3 -m dcas.scripts.crawl_itunes_previews \
    --out_dir "$OUT_DIR" \
    --countries "$country" \
    --target_total "$target" \
    --workers "$WORKERS" \
    --checkpoint_interval "$CHECKPOINT_INTERVAL" \
    --max_per_query "$MAX_PER_QUERY" \
    --resume
  status=$?
  echo "[COUNTRY END] $(date -Is) country=$country exit=$status"
  if [ "$status" -ne 0 ]; then
    echo "[STOP] crawler failed for country=$country"
    exit "$status"
  fi
done

python3 -m dcas.scripts.merge_metadata_dedup \
  --inputs "$OUT_DIR/metadata.csv" \
  --out ./storage/public/merged/metadata_merged.csv \
  --require_audio_exists

echo "[BALANCED CRAWL END] $(date -Is)"
