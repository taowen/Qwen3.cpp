param(
  [string]$OutputName = "",
  [string]$BuildDir = "",
  [string]$ExecuTorchRepoRoot = "",
  [string]$ConfigPath = "",
  [switch]$DryRun
)

$driver = Join-Path $PSScriptRoot "qwen3.ps1"
$forward = @{ Command = "export" }
if ($OutputName) { $forward.OutputName = $OutputName }
if ($BuildDir) { $forward.BuildDir = $BuildDir }
if ($ExecuTorchRepoRoot) { $forward.ExecuTorchRepoRoot = $ExecuTorchRepoRoot }
if ($ConfigPath) { $forward.ConfigPath = $ConfigPath }
if ($DryRun) { $forward.DryRun = $true }

& $driver @forward
exit $LASTEXITCODE
