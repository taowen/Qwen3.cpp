param(
  [ValidateSet("pull", "push")]
  [string]$Mode = "pull",
  [string]$ExecuTorchRoot = "",
  [string]$ConfigPath = "",
  [switch]$DryRun
)

$driver = Join-Path $PSScriptRoot "qwen3.ps1"
$forward = @{
  Command = "vendor-sync-vulkan"
  Mode = $Mode
}
if ($ExecuTorchRoot) { $forward.ExecuTorchRoot = $ExecuTorchRoot }
if ($ConfigPath) { $forward.ConfigPath = $ConfigPath }
if ($DryRun) { $forward.DryRun = $true }

& $driver @forward
exit $LASTEXITCODE
