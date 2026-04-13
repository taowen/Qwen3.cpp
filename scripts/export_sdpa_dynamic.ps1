param(
  [string]$OutputName = "qwen3_0_6b_vulkan_8da4w_kv_sdpa_dynamic.pte",
  [string]$BuildDir = "build/executorch-win-vulkan-llm-clangcl",
  [string]$ExecuTorchRepoRoot = "C:/Apps/qwen3-export"
)

$ErrorActionPreference = 'Stop'
$repoRoot = Resolve-Path (Join-Path $PSScriptRoot '..')
$upstreamRoot = Resolve-Path $ExecuTorchRepoRoot
Set-Location $upstreamRoot

$python = Join-Path $repoRoot '.venv\Scripts\python.exe'
if (!(Test-Path $python)) { throw "Missing python: $python. Run scripts/setup_env.ps1 first." }

$flatc = Join-Path $upstreamRoot ($BuildDir + '\third-party\flatc_ep\bin\flatc.exe')
if (!(Test-Path $flatc)) { throw "Missing flatc: $flatc" }

$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONPATH = (Join-Path $upstreamRoot 'third_party')
$env:FLATC_EXECUTABLE = $flatc

& $python -m executorch.examples.models.llama.export_llama `
  --model qwen3_0_6b `
  --params third_party/executorch/examples/models/qwen3/config/0_6b_config.json `
  -qmode 8da4w -G 128 -V -kv --use_sdpa_with_kv_cache `
  --output_name $OutputName

if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$src = Join-Path $upstreamRoot $OutputName
$dst = Join-Path $repoRoot ("runtime/models/{0}" -f $OutputName)
Copy-Item -Force $src $dst
Write-Host "Export done: $src"
Write-Host "Runtime model updated: $dst"
