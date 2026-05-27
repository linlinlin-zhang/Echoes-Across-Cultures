param(
    [int]$Port = 18011,
    [string]$HostName = "127.0.0.1",
    [string]$Protocol = "http2"
)

$ErrorActionPreference = "Stop"

$repoCloudflared = Join-Path (Get-Location) "tools\cloudflared.exe"
$commandCloudflared = Get-Command cloudflared -ErrorAction SilentlyContinue
if ($commandCloudflared) {
    $cloudflared = $commandCloudflared.Source
} elseif (Test-Path $repoCloudflared) {
    $cloudflared = $repoCloudflared
} else {
    throw "cloudflared was not found. Install it, put cloudflared.exe at tools\cloudflared.exe, or run scripts/run_cloudflare_quick_tunnel_wsl.sh from WSL."
}

$target = "http://$HostName`:$Port"
Write-Host "Starting Cloudflare Quick Tunnel for $target"
Write-Host "Copy the generated https://*.trycloudflare.com URL into ECHO_MAINLINE_WORKER_URL on the cloud server."
Write-Host "Using cloudflared: $cloudflared"

& $cloudflared tunnel --protocol $Protocol --url $target
