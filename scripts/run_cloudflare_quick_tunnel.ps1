param(
    [int]$Port = 18011,
    [string]$HostName = "127.0.0.1"
)

$ErrorActionPreference = "Stop"

$target = "http://$HostName`:$Port"
Write-Host "Starting Cloudflare Quick Tunnel for $target"
Write-Host "Copy the generated https://*.trycloudflare.com URL into ECHO_MAINLINE_WORKER_URL on the cloud server."

& cloudflared tunnel --url $target
