param(
  [string]$BuildDir = "build/qwen3-chatbot-clangcl",
  [string]$ModelPath = "",
  [string]$TokenizerPath = "",
  [int]$MaxNewTokens = 160,
  [double]$Temperature = 0,
  [int]$IgnoreEos = 1
)

$ErrorActionPreference = 'Stop'
$repoRoot = Resolve-Path (Join-Path $PSScriptRoot '..')
$exe = Join-Path $repoRoot ($BuildDir + '\Release\qwen3_chatbot.exe')
if (!(Test-Path $exe)) {
  throw "Missing chatbot exe: $exe. Run scripts/build_chatbot.ps1 first."
}

if ($ModelPath -eq '') {
  $ModelPath = Join-Path $repoRoot 'runtime\models\qwen3_0_6b_vulkan_8da4w_kv_sdpa_dynamic.pte'
}
if ($TokenizerPath -eq '') {
  $TokenizerPath = Join-Path $repoRoot 'runtime\tokenizer\tokenizer.json'
}

$args = @(
  '--model_path', $ModelPath,
  '--tokenizer_path', $TokenizerPath,
  '--max_new_tokens', $MaxNewTokens,
  '--temperature', $Temperature,
  '--ignore_eos', $IgnoreEos
)

& $exe @args
