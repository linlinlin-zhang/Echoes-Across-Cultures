param(
    [string]$SshHost = "echo-tunnel@47.250.196.219",
    [string]$SshFallbackHost = "echo-tunnel@100.120.12.57",
    [string]$SshKey = "$env:USERPROFILE\.ssh\echo_exhibition_ed25519",
    [string]$TaskName = "Echo-Intranet-Web",
    [switch]$Update
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repo ".venv-gpu\Scripts\python.exe"
$runner = Join-Path $repo "scripts\run_intranet_web.py"
if (-not (Test-Path $python)) { throw "Missing Python environment: $python" }
if (-not (Test-Path $SshKey)) { throw "Missing SSH private key: $SshKey" }
if (-not (Test-Path (Join-Path $repo "configs\local_web.env"))) {
    throw "Create configs/local_web.env from configs/local_web.env.example first."
}
if ((Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) -and -not $Update) {
    throw "Scheduled task already exists: $TaskName. Inspect it before replacing it."
}
$arguments = '"{0}" --ssh-host "{1}" --ssh-key "{2}" --ssh-fallback-host "{3}"' -f $runner, $SshHost, $SshKey, $SshFallbackHost
$action = New-ScheduledTaskAction -Execute $python -Argument $arguments -WorkingDirectory $repo
$user = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $user
$principal = New-ScheduledTaskPrincipal -UserId $user -LogonType Interactive -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -StartWhenAvailable -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 1) -MultipleInstances IgnoreNew
Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger `
    -Principal $principal -Settings $settings -Force:$Update -Description "Echo website and reconnecting SSH tunnel to cloud Nginx" | Out-Null
Start-ScheduledTask -TaskName $TaskName
Get-ScheduledTask -TaskName $TaskName | Select-Object TaskName, State
