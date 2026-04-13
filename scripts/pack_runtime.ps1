param(
  [string]$Model = "qwen3_0_6b_vulkan_8da4w_kv_sdpa_dynamic.pte",
  [string]$RunnerBuildDir = "build/executorch-win-vulkan-llm-clangcl",
  [string]$TokenizerPath = "C:\Users\taowen\.cache\huggingface\hub\models--Qwen--Qwen3-0.6B\snapshots\c1899de289a04d12100db370d81485cdf75e47ca\tokenizer.json",
  [string]$ExecuTorchRepoRoot = "C:/Apps/qwen3-export"
)

$ErrorActionPreference = 'Stop'
$repoRoot = Resolve-Path (Join-Path $PSScriptRoot '..')
$upstreamRoot = Resolve-Path $ExecuTorchRepoRoot
$runtimeRoot = Resolve-Path (Join-Path $PSScriptRoot '..\runtime')

$runner = Join-Path $upstreamRoot ($RunnerBuildDir + '\examples\models\llama\Release\llama_main.exe')
$modelSrc = Join-Path $upstreamRoot $Model
$modelDst = Join-Path $runtimeRoot ("models\{0}" -f [IO.Path]::GetFileName($Model))
$tokDst = Join-Path $runtimeRoot 'tokenizer\tokenizer.json'
$promptSrc = Join-Path $upstreamRoot 'perf_prompt.txt'
$promptDst = Join-Path $runtimeRoot 'prompts\perf_prompt.txt'

if (!(Test-Path $runner)) { throw "Missing runner: $runner" }
if (!(Test-Path $modelSrc)) { throw "Missing model: $modelSrc" }
if (!(Test-Path $TokenizerPath)) { throw "Missing tokenizer: $TokenizerPath" }
if (!(Test-Path $promptSrc)) { throw "Missing prompt: $promptSrc" }

Copy-Item -Force $runner (Join-Path $runtimeRoot 'bin\llama_main.exe')
Copy-Item -Force $modelSrc $modelDst
Copy-Item -Force $TokenizerPath $tokDst
Copy-Item -Force $promptSrc $promptDst

Write-Host "Runtime bundle updated: $runtimeRoot"
