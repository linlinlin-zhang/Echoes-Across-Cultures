Set-Location $PSScriptRoot

Start-Process -FilePath "python" -ArgumentList "-m", "dcas_server" -WorkingDirectory $PSScriptRoot
Write-Host "DCAS API started."
Write-Host "Current prototype frontend is the static web/ directory mounted by FastAPI."
Write-Host "Open http://localhost:8000/ or http://localhost:8000/music.html."
