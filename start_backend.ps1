Set-Location "$PSScriptRoot"
if (-not (Test-Path "backend/.venv/Scripts/Activate.ps1")) { Write-Host "Create backend venv and install backend requirements first."; exit 1 }
. "backend/.venv/Scripts/Activate.ps1"
python -m uvicorn app.main:app --app-dir backend --reload --port 8000
