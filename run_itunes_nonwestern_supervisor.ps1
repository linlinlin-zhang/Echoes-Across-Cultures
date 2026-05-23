# Self-checking iTunes non-Western culture top-up supervisor.
#
# Default targets:
#   china=700
#   japan/korea/india/brazil/latin/africa/middle_east/southeast_asia=500

$OUT_DIR = "./storage/public/itunes_crawl"
$TARGETS = "china=700,japan=500,korea=500,india=500,brazil=500,latin=500,africa=500,middle_east=500,southeast_asia=500"
$WORKERS = 4
$CHECKPOINT_INTERVAL = 300
$MAX_PER_QUERY = 50
$BATCH_SIZE = 120
$RESTART_DELAY = 120
$IDLE_ROUND_LIMIT = 8

Write-Host "[START] Self-checking iTunes non-Western supervisor" -ForegroundColor Green
Write-Host "  Output:  $OUT_DIR"
Write-Host "  Targets: $TARGETS"

python -m dcas.scripts.supervise_itunes_culture_crawl `
  --out_dir $OUT_DIR `
  --targets $TARGETS `
  --workers $WORKERS `
  --checkpoint_interval $CHECKPOINT_INTERVAL `
  --max_per_query $MAX_PER_QUERY `
  --batch_size $BATCH_SIZE `
  --restart_delay $RESTART_DELAY `
  --idle_round_limit $IDLE_ROUND_LIMIT

Write-Host "[DONE] Supervisor exited." -ForegroundColor Green
