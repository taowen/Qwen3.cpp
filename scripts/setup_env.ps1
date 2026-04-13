param(
  [string]$PythonVersion = "3.11",
  [switch]$Recreate
)

$ErrorActionPreference = 'Stop'
$repoRoot = Resolve-Path (Join-Path $PSScriptRoot '..')
Set-Location $repoRoot

if ($Recreate -and (Test-Path '.venv')) {
  Remove-Item -Recurse -Force '.venv'
}

uv venv .venv --python $PythonVersion
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

uv sync
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Environment ready: $repoRoot\.venv"
