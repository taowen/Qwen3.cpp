param(
  [string]$Model = "",
  [int]$Runs = 0,
  [int]$MaxNewTokens = 0,
  [string]$PromptFile = "",
  [string]$ConfigPath = "",
  [switch]$DryRun
)

$driver = Join-Path $PSScriptRoot "qwen3.ps1"
$forward = @{ Command = "bench" }
if ($Model) { $forward.Model = $Model }
if ($Runs -gt 0) { $forward.Runs = $Runs }
if ($MaxNewTokens -gt 0) { $forward.MaxNewTokens = $MaxNewTokens }
if ($PromptFile) { $forward.PromptFile = $PromptFile }
if ($ConfigPath) { $forward.ConfigPath = $ConfigPath }
if ($DryRun) { $forward.DryRun = $true }

& $driver @forward
exit $LASTEXITCODE
