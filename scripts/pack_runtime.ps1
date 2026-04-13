param(
  [string]$Model = "",
  [string]$RunnerBuildDir = "",
  [string]$TokenizerPath = "",
  [string]$ExecuTorchRepoRoot = "",
  [string]$ConfigPath = "",
  [switch]$DryRun
)

$driver = Join-Path $PSScriptRoot "qwen3.ps1"
$forward = @{ Command = "pack" }
if ($Model) { $forward.Model = $Model }
if ($RunnerBuildDir) { $forward.BuildDir = $RunnerBuildDir }
if ($TokenizerPath) { $forward.TokenizerPath = $TokenizerPath }
if ($ExecuTorchRepoRoot) { $forward.ExecuTorchRepoRoot = $ExecuTorchRepoRoot }
if ($ConfigPath) { $forward.ConfigPath = $ConfigPath }
if ($DryRun) { $forward.DryRun = $true }

& $driver @forward
exit $LASTEXITCODE
