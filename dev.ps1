Set-Location $PSScriptRoot

Start-Process -FilePath "python" -ArgumentList "-m", "dcas_server" -WorkingDirectory $PSScriptRoot
Write-Host "DCAS API started."
Write-Host "Current prototype frontend is the static web_prototype/ directory."
Write-Host "The retired web/ development server is no longer launched by this script."
