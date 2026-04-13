param(
  [string]$ConfigPath = "",
  [switch]$SkipExternal,
  [switch]$SkipRuntimeArtifacts
)

$driver = Join-Path $PSScriptRoot "qwen3.ps1"
$forward = @{ Command = "doctor" }
if ($ConfigPath) { $forward.ConfigPath = $ConfigPath }
if ($SkipExternal) { $forward.SkipExternal = $true }
if ($SkipRuntimeArtifacts) { $forward.SkipRuntimeArtifacts = $true }

& $driver @forward
exit $LASTEXITCODE
