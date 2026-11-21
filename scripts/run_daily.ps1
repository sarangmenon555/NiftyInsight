$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
New-Item -ItemType Directory -Force -Path logs | Out-Null
$python = Join-Path (Get-Location) ".venv\Scripts\python.exe"
& $python -m src.pipeline run-daily *>> logs\run_daily.log
git add reports
git diff --cached --quiet
if ($LASTEXITCODE -ne 0) {
    git commit -m "Update signals $(Get-Date -Format yyyy-MM-dd)"
    git push
}
