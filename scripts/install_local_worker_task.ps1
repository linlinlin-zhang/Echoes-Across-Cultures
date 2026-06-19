param(
    [string]$TaskName = "Echo Local Mainline Worker",
    [int]$Port = 18011,
    [string]$HostName = "127.0.0.1",
    [string]$EnvFile = "configs/local_worker.env"
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$scriptPath = Join-Path $repoRoot "scripts\run_local_mainline_worker.ps1"

if (-not (Test-Path $scriptPath)) {
    throw "Missing script: $scriptPath"
}

$argument = @(
    "-NoProfile",
    "-ExecutionPolicy Bypass",
    "-File `"$scriptPath`"",
    "-Port $Port",
    "-HostName `"$HostName`"",
    "-EnvFile `"$EnvFile`""
) -join " "

$action = New-ScheduledTaskAction `
    -Execute "powershell.exe" `
    -Argument $argument `
    -WorkingDirectory $repoRoot

$trigger = New-ScheduledTaskTrigger -AtLogOn
$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 1)

try {
    Register-ScheduledTask `
        -TaskName $TaskName `
        -Action $action `
        -Trigger $trigger `
        -Settings $settings `
        -Description "Run Echo's model-heavy local recommendation worker on this Windows host." `
        -Force `
        -ErrorAction Stop | Out-Null

    Write-Host "Installed scheduled task: $TaskName"
    Write-Host "Start it now with:"
    Write-Host "  Start-ScheduledTask -TaskName `"$TaskName`""
    exit 0
} catch {
    Write-Warning "Scheduled task install failed: $($_.Exception.Message)"
    Write-Warning "Falling back to a current-user Startup shortcut."
}

$startupDir = [Environment]::GetFolderPath("Startup")
$shortcutPath = Join-Path $startupDir "$TaskName.lnk"
$shell = New-Object -ComObject WScript.Shell
$shortcut = $shell.CreateShortcut($shortcutPath)
$shortcut.TargetPath = "powershell.exe"
$shortcut.Arguments = $argument
$shortcut.WorkingDirectory = $repoRoot
$shortcut.WindowStyle = 7
$shortcut.Description = "Run Echo's model-heavy local recommendation worker on this Windows host."
$shortcut.Save()

Write-Host "Installed Startup shortcut: $shortcutPath"
Write-Host "Start Echo worker now with:"
Write-Host "  powershell.exe $argument"
