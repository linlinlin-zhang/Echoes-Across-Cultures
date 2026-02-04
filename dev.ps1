Set-Location $PSScriptRoot

Start-Process -FilePath "python" -ArgumentList "-m", "dcas_server" -WorkingDirectory $PSScriptRoot
Start-Process -FilePath "npm" -ArgumentList "run", "dev", "--", "--host", "0.0.0.0", "--port", "5173" -WorkingDirectory (Join-Path $PSScriptRoot "web")

