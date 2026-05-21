# Jamendo CC-Licensed Music Crawler Launch Script for DCAS
# Usage: Fill in your Jamendo Client ID below, then run: .\run_jamendo_crawl.ps1
#
# To get your credentials:
#   1. Go to https://devportal.jamendo.com/
#   2. Sign up and create an API client
#   3. Copy your Client ID

# ---------------------------------------------------------------------------
# FILL IN YOUR CREDENTIALS HERE
# ---------------------------------------------------------------------------
$JAMENDO_CLIENT_ID = "<YOUR_JAMENDO_CLIENT_ID>"

# ---------------------------------------------------------------------------
# Validate
# ---------------------------------------------------------------------------
if ($JAMENDO_CLIENT_ID -eq "<YOUR_JAMENDO_CLIENT_ID>") {
    Write-Host "[ERROR] Please edit this file and set your Jamendo Client ID." -ForegroundColor Red
    exit 1
}

# ---------------------------------------------------------------------------
# Crawl configuration
# ---------------------------------------------------------------------------
$OUT_DIR            = "./storage/public/jamendo_crawl"
$TARGET_TOTAL       = 20000           # Total unique tracks to collect
$WORKERS            = 6                # Parallel download workers
$CHECKPOINT_INTERVAL = 300             # Seconds between state saves

# Optional: limit to specific cultures (leave empty for all)
# Valid cultures: west, china, korea, japan, india, latin, brazil, africa, middle_east, southeast_asia, celtic
# Example: $CULTURES = "china,korea,japan,india"
$CULTURES = ""

# ---------------------------------------------------------------------------
# Build argument list
# ---------------------------------------------------------------------------
$argList = @(
    "-m", "dcas.scripts.crawl_jamendo",
    "--client_id", $JAMENDO_CLIENT_ID,
    "--out_dir", $OUT_DIR,
    "--target_total", $TARGET_TOTAL,
    "--workers", $WORKERS,
    "--checkpoint_interval", $CHECKPOINT_INTERVAL
)

if ($CULTURES -ne "") {
    $argList += @("--cultures", $CULTURES)
}

# Check if a previous crawl exists -> auto-resume
$stateFile = Join-Path $OUT_DIR "state.json"
if (Test-Path $stateFile) {
    Write-Host "[INFO] Found existing state.json -> auto-resuming crawl." -ForegroundColor Cyan
    $argList += "--resume"
}

# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------
Write-Host "[START] Jamendo crawler launching..." -ForegroundColor Green
Write-Host "  Target:      $TARGET_TOTAL tracks"
Write-Host "  Workers:     $WORKERS"
Write-Host "  Output:      $OUT_DIR"
Write-Host ""

& python @argList

Write-Host ""
Write-Host "[DONE] Crawl finished. Check $OUT_DIR for results." -ForegroundColor Green
