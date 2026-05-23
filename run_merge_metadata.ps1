# Merge available crawler metadata into a unified metadata_merged.csv
# Usage: After iTunes and/or Jamendo crawls finish, run: .\run_merge_metadata.ps1

# ---------------------------------------------------------------------------
# Input paths (adjust if your crawl output directories differ)
# ---------------------------------------------------------------------------
$ITUNES_METADATA  = "./storage/public/itunes_crawl/metadata.csv"
$JAMENDO_METADATA = "./storage/public/jamendo_crawl/metadata.csv"
$SPOTIFY_METADATA = "./storage/public/spotify_crawl/metadata.csv"
$OUTPUT_DIR       = "./storage/public/merged"
$OUTPUT_FILE      = "$OUTPUT_DIR/metadata_merged.csv"

# ---------------------------------------------------------------------------
# Optional: require that audio files actually exist on disk
# (Enable this to filter out rows whose MP3 preview was lost/deleted)
# ---------------------------------------------------------------------------
$REQUIRE_AUDIO_EXISTS = $false

# ---------------------------------------------------------------------------
# Collect available inputs
# ---------------------------------------------------------------------------
$inputs = @()
foreach ($candidate in @($ITUNES_METADATA, $JAMENDO_METADATA, $SPOTIFY_METADATA)) {
    if (Test-Path $candidate) {
        $inputs += $candidate
    }
}

if ($inputs.Count -eq 0) {
    Write-Host "[ERROR] No crawler metadata files found." -ForegroundColor Red
    Write-Host "Expected one or more of:" -ForegroundColor Yellow
    Write-Host "  - $ITUNES_METADATA"
    Write-Host "  - $JAMENDO_METADATA"
    Write-Host "  - $SPOTIFY_METADATA"
    exit 1
}

# Ensure output directory exists
New-Item -ItemType Directory -Force -Path $OUTPUT_DIR | Out-Null

# ---------------------------------------------------------------------------
# Build argument list
# ---------------------------------------------------------------------------
$argList = @(
    "-m", "dcas.scripts.merge_metadata_dedup",
    "--inputs"
)
$argList += $inputs
$argList += @(
    "--out", $OUTPUT_FILE
)

if ($REQUIRE_AUDIO_EXISTS) {
    $argList += "--require_audio_exists"
}

# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------
Write-Host "[START] Merging metadata..." -ForegroundColor Green
Write-Host "  Inputs:"
foreach ($inputPath in $inputs) { Write-Host "    - $inputPath" }
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
