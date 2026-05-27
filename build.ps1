Set-Location $PSScriptRoot

python -m pip install -r requirements.txt
Write-Host "Python dependencies installed."
Write-Host "The legacy npm/Vite build pipeline has been retired."
Write-Host "Current prototype assets live in the static web/ directory and are mounted by FastAPI."
