param(
    [int]$Port = 18011,
    [string]$HostName = "127.0.0.1",
    [string]$EnvFile = "configs/local_worker.env"
)

$ErrorActionPreference = "Stop"

if (Test-Path $EnvFile) {
    Get-Content $EnvFile | ForEach-Object {
        $line = $_.Trim()
        if (-not $line -or $line.StartsWith("#") -or -not $line.Contains("=")) {
            return
        }
        $name, $value = $line.Split("=", 2)
        [Environment]::SetEnvironmentVariable($name.Trim(), $value.Trim(), "Process")
    }
} else {
    Write-Host "Env file not found: $EnvFile"
    Write-Host "Copy configs/local_worker.env.example to configs/local_worker.env and set ECHO_WORKER_SHARED_TOKEN."
}

if (-not $env:ECHO_WORKER_REQUIRE_TOKEN) {
    $env:ECHO_WORKER_REQUIRE_TOKEN = "true"
}

if (-not $env:ECHO_WORKER_SHARED_TOKEN -or $env:ECHO_WORKER_SHARED_TOKEN -eq "change-this-to-a-long-random-secret") {
    Write-Warning "ECHO_WORKER_SHARED_TOKEN is not set to a private value. Do not expose this worker publicly yet."
}

$pythonCandidates = @(
    ".\.venv-gpu\Scripts\python.exe",
    ".\.venv\Scripts\python.exe",
    "python"
)
$python = $null
foreach ($candidate in $pythonCandidates) {
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
}

if (-not $python) {
    throw "No usable Python was found. Create .venv-gpu, .venv, or install Python first."
}

Write-Host "Starting Echo local mainline worker at http://$HostName`:$Port"
& $python -m uvicorn dcas_server.app:app --host $HostName --port $Port --proxy-headers
