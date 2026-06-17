param(
    [int]$Port = 8000,
    [string]$HostName = "0.0.0.0",
    [string]$EnvFile = "configs/public_host.env"
)

$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $repoRoot

function Import-DotEnv {
    param([string]$Path)

    if (-not (Test-Path $Path)) {
        return
    }

    Get-Content $Path | ForEach-Object {
        $line = $_.Trim()
        if (-not $line -or $line.StartsWith("#") -or -not $line.Contains("=")) {
            return
        }
        $name, $value = $line.Split("=", 2)
        [Environment]::SetEnvironmentVariable($name.Trim(), $value.Trim(), "Process")
    }
}

if (Test-Path $EnvFile) {
    Import-DotEnv $EnvFile
} else {
    $example = "configs/public_host.env.example"
    Write-Warning "Env file not found: $EnvFile. Loading $example instead."
    Import-DotEnv $example
}

if (-not $env:ECHO_STORAGE_ROOT) {
    $env:ECHO_STORAGE_ROOT = "storage"
}
if (-not $env:ECHO_USER_DATA_DIR) {
    $env:ECHO_USER_DATA_DIR = "storage/user_data"
}

$wingetLinks = Join-Path $env:LOCALAPPDATA "Microsoft\WinGet\Links"
if ((Test-Path $wingetLinks) -and -not ($env:Path -split ";" | Where-Object { $_ -ieq $wingetLinks })) {
    $env:Path = "$wingetLinks;$env:Path"
}

$pythonCandidates = @(
    ".\.venv\Scripts\python.exe",
    ".\.venv-gpu\Scripts\python.exe",
    "python"
)
$python = $null
foreach ($candidate in $pythonCandidates) {
    try {
        if ($candidate -eq "python") {
            $cmd = Get-Command python -ErrorAction SilentlyContinue
            if ($cmd) {
                $python = $cmd.Source
                break
            }
        } elseif (Test-Path $candidate) {
            $python = $candidate
            break
        }
    } catch {
        continue
    }
}

if (-not $python) {
    throw "No usable Python was found. Create .venv or install Python first."
}

New-Item -ItemType Directory -Force -Path "logs" | Out-Null

Write-Host "Starting Echo public host at http://$HostName`:$Port"
Write-Host "Repository: $repoRoot"
Write-Host "Python: $python"
Write-Host "Env file: $EnvFile"

& $python -m uvicorn dcas_server.app:app `
    --host $HostName `
    --port $Port `
    --proxy-headers `
    "--forwarded-allow-ips=*"
