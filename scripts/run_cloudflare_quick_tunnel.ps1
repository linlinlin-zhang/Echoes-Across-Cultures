param(
    [int]$Port = 18011,
    [string]$HostName = "127.0.0.1",
    [string]$Protocol = "http2",
    [switch]$UseWsl
)

$ErrorActionPreference = "Stop"

function Test-CloudflaredExecutable {
    param([string]$Path)
    try {
        $null = & $Path --version 2>$null
        return $LASTEXITCODE -eq 0
    } catch {
        return $false
    }
}

function Convert-WindowsPathToWslPath {
    param([string]$Path)
    $resolved = (Resolve-Path -LiteralPath $Path).Path
    if ($resolved -match "^([A-Za-z]):\\(.*)$") {
        $drive = $Matches[1].ToLowerInvariant()
        $rest = $Matches[2] -replace "\\", "/"
        return "/mnt/$drive/$rest"
    }
    throw "Cannot convert Windows path to WSL path: $resolved"
}

function Invoke-WslTunnel {
    $wsl = Get-Command wsl.exe -ErrorAction SilentlyContinue
    if (-not $wsl) {
        throw "No usable Windows cloudflared was found, and wsl.exe is unavailable. Install cloudflared for Windows or run scripts/run_cloudflare_quick_tunnel_wsl.sh from WSL."
    }

    $repoWinPath = (Get-Location).Path
    $repoWslPath = Convert-WindowsPathToWslPath $repoWinPath
    if (-not $repoWslPath) {
        throw "Could not convert repository path for WSL: $repoWinPath"
    }

    $envParts = @(
        "PORT='$Port'",
        "PROTOCOL='$Protocol'"
    )
    if ($HostName -and $HostName -notin @("127.0.0.1", "localhost")) {
        $envParts += "HOST_NAME='$HostName'"
    } else {
        Write-Warning "Using WSL fallback. If the worker runs on Windows, start it with -HostName 0.0.0.0 so WSL can reach it."
    }

    $command = "cd '$repoWslPath' && $($envParts -join ' ') ./scripts/run_cloudflare_quick_tunnel_wsl.sh"
    Write-Host "Starting Cloudflare Quick Tunnel through WSL..."
    Write-Host "WSL command: $command"
    & $wsl.Source bash -lc $command
    exit $LASTEXITCODE
}

$repoCloudflared = Join-Path (Get-Location) "tools\cloudflared.exe"
$commandCloudflared = Get-Command cloudflared -ErrorAction SilentlyContinue
if (-not $UseWsl) {
    $candidates = @()
    if ($commandCloudflared) {
        $candidates += $commandCloudflared.Source
    }
    if (Test-Path $repoCloudflared) {
        $candidates += $repoCloudflared
    }

    foreach ($candidate in $candidates) {
        if (Test-CloudflaredExecutable $candidate) {
            $cloudflared = $candidate
            break
        }
        Write-Warning "Ignoring unusable cloudflared executable: $candidate"
    }
}

if (-not $cloudflared) {
    Invoke-WslTunnel
}

$target = "http://$HostName`:$Port"
Write-Host "Starting Cloudflare Quick Tunnel for $target"
Write-Host "Copy the generated https://*.trycloudflare.com URL into ECHO_MAINLINE_WORKER_URL on the cloud server."
Write-Host "Using cloudflared: $cloudflared"

& $cloudflared tunnel --protocol $Protocol --url $target
