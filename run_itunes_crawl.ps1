# Apple iTunes Search API Preview Crawler Launch Script
# No API key required. Rate limit: ~20 req/min.

$OUT_DIR            = "./storage/public/itunes_crawl"
$TARGET_TOTAL       = 20000
$WORKERS            = 4
$CHECKPOINT_INTERVAL = 300
$MAX_PER_QUERY      = 50

# Optional: limit countries (empty = all ~50)
# $COUNTRIES = "US,JP,KR,GB,BR,MX,IN,FR,DE"
$COUNTRIES = ""

$argList = @(
    "-m", "dcas.scripts.crawl_itunes_previews",
    "--out_dir", $OUT_DIR,
    "--target_total", $TARGET_TOTAL,
    "--workers", $WORKERS,
    "--checkpoint_interval", $CHECKPOINT_INTERVAL,
    "--max_per_query", $MAX_PER_QUERY
)

if ($COUNTRIES -ne "") {
    $argList += @("--countries", $COUNTRIES)
}

$stateFile = Join-Path $OUT_DIR "state.json"
if (Test-Path $stateFile) {
    Write-Host "[INFO] Found existing state.json -> auto-resuming." -ForegroundColor Cyan
    $argList += "--resume"
}

Write-Host "[START] iTunes crawler launching..." -ForegroundColor Green
& python @argList
Write-Host "[DONE] Check $OUT_DIR for results." -ForegroundColor Green
