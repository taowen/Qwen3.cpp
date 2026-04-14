param(
  [string]$ConfigPath = "",
  [int]$Runs = 0,
  [int]$MaxNewTokens = 0,
  [double]$Temperature = [double]::NaN,
  [string]$PromptFile = "",
  [string]$OutDir = ""
)

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$python = Join-Path $repoRoot ".venv\Scripts\python.exe"
if (!(Test-Path $python)) {
  throw "Missing virtualenv python: $python"
}

$script = Join-Path $PSScriptRoot "vk_perf_compare.py"
$args = @($script)
if ($ConfigPath) { $args += @("--config-path", $ConfigPath) }
if ($Runs -gt 0) { $args += @("--runs", [string]$Runs) }
if ($MaxNewTokens -gt 0) { $args += @("--max-new-tokens", [string]$MaxNewTokens) }
if (![double]::IsNaN($Temperature)) { $args += @("--temperature", [string]$Temperature) }
if ($PromptFile) { $args += @("--prompt-file", $PromptFile) }
if ($OutDir) { $args += @("--out-dir", $OutDir) }

& $python @args
exit $LASTEXITCODE
