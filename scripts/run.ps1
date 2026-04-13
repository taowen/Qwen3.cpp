param(
  [string]$Model = "",
  [int]$MaxNewTokens = 0,
  [double]$Temperature = [double]::NaN,
  [switch]$IgnoreEos,
  [switch]$NoIgnoreEos,
  [string]$PromptFile = "",
  [string]$Prompt = "",
  [string]$ConfigPath = "",
  [switch]$DryRun
)

$driver = Join-Path $PSScriptRoot "qwen3.ps1"
$forward = @{ Command = "run" }
if ($Model) { $forward.Model = $Model }
if ($MaxNewTokens -gt 0) { $forward.MaxNewTokens = $MaxNewTokens }
if (![double]::IsNaN($Temperature)) { $forward.Temperature = $Temperature }
if ($IgnoreEos) { $forward.IgnoreEos = $true }
if ($NoIgnoreEos) { $forward.NoIgnoreEos = $true }
if ($PromptFile) { $forward.PromptFile = $PromptFile }
if ($Prompt) { $forward.Prompt = $Prompt }
if ($ConfigPath) { $forward.ConfigPath = $ConfigPath }
if ($DryRun) { $forward.DryRun = $true }

& $driver @forward
exit $LASTEXITCODE
