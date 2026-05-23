Set-Location $PSScriptRoot

python -m pip install -r requirements.txt
Write-Host "Python dependencies installed."
Write-Host "The legacy web/ build pipeline has been retired."
Write-Host "Current prototype assets live in web_prototype/ and do not require npm build here."
