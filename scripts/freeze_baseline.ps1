param(
  [string]$ConfigPath = "",
  [switch]$DryRun
)

$driver = Join-Path $PSScriptRoot "qwen3.ps1"
$forward = @{ Command = "baseline" }
if ($ConfigPath) { $forward.ConfigPath = $ConfigPath }
if ($DryRun) { $forward.DryRun = $true }

& $driver @forward
exit $LASTEXITCODE
