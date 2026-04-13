param(
  [string]$BuildDir = "",
  [string]$Model = "",
  [string]$TokenizerPath = "",
  [int]$MaxNewTokens = 0,
  [double]$Temperature = [double]::NaN,
  [switch]$IgnoreEos,
  [switch]$NoIgnoreEos,
  [string]$ConfigPath = "",
  [switch]$DryRun
)

$driver = Join-Path $PSScriptRoot "qwen3.ps1"
$forward = @{ Command = "chatbot-run" }
if ($BuildDir) { $forward.BuildDir = $BuildDir }
if ($Model) { $forward.Model = $Model }
if ($TokenizerPath) { $forward.TokenizerPath = $TokenizerPath }
if ($MaxNewTokens -gt 0) { $forward.MaxNewTokens = $MaxNewTokens }
if (![double]::IsNaN($Temperature)) { $forward.Temperature = $Temperature }
if ($IgnoreEos) { $forward.IgnoreEos = $true }
if ($NoIgnoreEos) { $forward.NoIgnoreEos = $true }
if ($ConfigPath) { $forward.ConfigPath = $ConfigPath }
if ($DryRun) { $forward.DryRun = $true }

& $driver @forward
exit $LASTEXITCODE
