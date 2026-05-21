# Merge Spotify + Jamendo metadata into a unified metadata_merged.csv
# Usage: After both crawls finish, run: .\run_merge_metadata.ps1

# ---------------------------------------------------------------------------
# Input paths (adjust if your crawl output directories differ)
# ---------------------------------------------------------------------------
$SPOTIFY_METADATA = "./storage/public/spotify_crawl/metadata.csv"
$JAMENDO_METADATA = "./storage/public/jamendo_crawl/metadata.csv"
$OUTPUT_DIR       = "./storage/public/merged"
$OUTPUT_FILE      = "$OUTPUT_DIR/metadata_merged.csv"

# ---------------------------------------------------------------------------
# Optional: require that audio files actually exist on disk
# (Enable this to filter out rows whose MP3 preview was lost/deleted)
# ---------------------------------------------------------------------------
$REQUIRE_AUDIO_EXISTS = $false

# ---------------------------------------------------------------------------
# Validate inputs exist
# ---------------------------------------------------------------------------
$missing = @()
if (-not (Test-Path $SPOTIFY_METADATA)) { $missing += $SPOTIFY_METADATA }
if (-not (Test-Path $JAMENDO_METADATA)) { $missing += $JAMENDO_METADATA }

if ($missing.Count -gt 0) {
    Write-Host "[ERROR] Missing input file(s):" -ForegroundColor Red
    foreach ($m in $missing) { Write-Host "  - $m" }
    Write-Host ""
    Write-Host "Please ensure both crawls have completed and generated metadata.csv." -ForegroundColor Yellow
    exit 1
}

# Ensure output directory exists
New-Item -ItemType Directory -Force -Path $OUTPUT_DIR | Out-Null

# ---------------------------------------------------------------------------
# Build argument list
# ---------------------------------------------------------------------------
$argList = @(
    "-m", "dcas.scripts.merge_spotify_jamendo_metadata",
    "--spotify", $SPOTIFY_METADATA,
    "--jamendo", $JAMENDO_METADATA,
    "--out", $OUTPUT_FILE
)

if ($REQUIRE_AUDIO_EXISTS) {
    $argList += "--require_audio_exists"
}

# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------
Write-Host "[START] Merging metadata..." -ForegroundColor Green
Write-Host "  Spotify: $SPOTIFY_METADATA"
Write-Host "  Jamendo: $JAMENDO_METADATA"
Write-Host "  Output:  $OUTPUT_FILE"
Write-Host ""

& python @argList

Write-Host ""
Write-Host "[NEXT STEP] Generate CultureMERT embeddings:" -ForegroundColor Cyan
Write-Host "  python -m dcas.scripts.build_tracks_from_audio `\"
Write-Host "    --metadata $OUTPUT_FILE `\"
Write-Host "    --out $OUTPUT_DIR/tracks.npz `\"
Write-Host "    --model_id ntua-slp/CultureMERT-95M `\"
Write-Host "    --pooling mean `\"
Write-Host "    --max_seconds 30 `\"
Write-Host "    --skip_errors"
