param(
    [int]$Port = 8000,
    [string]$RuleName = "Echo Public Host - Tailscale only"
)

$ErrorActionPreference = "Stop"

$existing = Get-NetFirewallRule -DisplayName $RuleName -ErrorAction SilentlyContinue
if ($existing) {
    Remove-NetFirewallRule -DisplayName $RuleName -ErrorAction Stop
}

New-NetFirewallRule `
    -DisplayName $RuleName `
    -Direction Inbound `
    -Action Allow `
    -Protocol TCP `
    -LocalPort $Port `
    -RemoteAddress "100.64.0.0/10","fd7a:115c:a1e0::/48" `
    -Profile Any `
    -ErrorAction Stop | Out-Null

Write-Host "Firewall rule installed: $RuleName"
Write-Host "Allowed inbound TCP $Port from Tailscale IPv4/IPv6 ranges only."
