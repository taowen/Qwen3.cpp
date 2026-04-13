param(
  [string]$Model = "qwen3_0_6b_vulkan_8da4w_kv_sdpa_dynamic.pte",
  [int]$MaxNewTokens = 80,
  [double]$Temperature = 0,
  [switch]$IgnoreEos = $true,
  [string]$PromptFile = "",
  [string]$Prompt = ""
)

$ErrorActionPreference = 'Stop'
$runtimeRoot = Resolve-Path (Join-Path $PSScriptRoot '..\runtime')
$exe = Join-Path $runtimeRoot 'bin\llama_main.exe'
$modelPath = Join-Path $runtimeRoot ("models\{0}" -f $Model)
$tokenizerPath = Join-Path $runtimeRoot 'tokenizer\tokenizer.json'

if (!(Test-Path $exe)) { throw "Missing runner: $exe" }
if (!(Test-Path $modelPath)) { throw "Missing model: $modelPath" }
if (!(Test-Path $tokenizerPath)) { throw "Missing tokenizer: $tokenizerPath" }

$args = @(
  '--model_path', $modelPath,
  '--tokenizer_path', $tokenizerPath,
  '--temperature', $Temperature,
  '--max_new_tokens', $MaxNewTokens
)

if ($IgnoreEos) { $args += '--ignore_eos' }

if ($PromptFile -ne '') {
  $args += @('--prompt_file', $PromptFile)
} elseif ($Prompt -ne '') {
  $args += @('--prompt', $Prompt)
} else {
  $defaultPrompt = Join-Path $runtimeRoot 'prompts\perf_prompt.txt'
  $args += @('--prompt_file', $defaultPrompt)
}

& $exe @args
exit $LASTEXITCODE
