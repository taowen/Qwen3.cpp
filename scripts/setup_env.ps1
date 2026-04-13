param(
  [string]$PythonVersion = "",
  [switch]$Recreate,
  [string]$ConfigPath = "",
  [switch]$DryRun
)

$driver = Join-Path $PSScriptRoot "qwen3.ps1"
$forward = @{ Command = "env" }
if ($PythonVersion) { $forward.PythonVersion = $PythonVersion }
if ($Recreate) { $forward.Recreate = $true }
if ($ConfigPath) { $forward.ConfigPath = $ConfigPath }
if ($DryRun) { $forward.DryRun = $true }

& $driver @forward
exit $LASTEXITCODE
