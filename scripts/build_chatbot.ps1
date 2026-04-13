param(
  [string]$BuildDir = "",
  [string]$ExecuTorchPrefix = "",
  [string]$ConfigPath = "",
  [switch]$DryRun
)

$driver = Join-Path $PSScriptRoot "qwen3.ps1"
$forward = @{ Command = "chatbot-build" }
if ($BuildDir) { $forward.BuildDir = $BuildDir }
if ($ExecuTorchPrefix) { $forward.ExecuTorchPrefix = $ExecuTorchPrefix }
if ($ConfigPath) { $forward.ConfigPath = $ConfigPath }
if ($DryRun) { $forward.DryRun = $true }

& $driver @forward
exit $LASTEXITCODE
