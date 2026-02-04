Set-Location $PSScriptRoot

python -m pip install -r requirements.txt
Push-Location (Join-Path $PSScriptRoot "web")
npm install
npm run build
Pop-Location

