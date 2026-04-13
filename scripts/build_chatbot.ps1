param(
  [string]$BuildDir = "build/qwen3-chatbot-clangcl",
  [string]$ExecuTorchPrefix = "C:/Apps/qwen3-export/build/executorch-win-vulkan-llm-clangcl/install",
  [string]$Generator = "Visual Studio 17 2022",
  [string]$Toolset = "ClangCL"
)

$ErrorActionPreference = 'Stop'
$repoRoot = Resolve-Path (Join-Path $PSScriptRoot '..')
$srcDir = Join-Path $repoRoot 'vendor-runtime'
$binDir = Join-Path $repoRoot $BuildDir

if (!(Test-Path $ExecuTorchPrefix)) {
  throw "EXECUTORCH_PREFIX not found: $ExecuTorchPrefix"
}

$configureArgs = @(
  '-S', $srcDir,
  '-B', $binDir,
  '-G', $Generator,
  '-T', $Toolset,
  "-DEXECUTORCH_PREFIX=$ExecuTorchPrefix"
)

cmake @configureArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

cmake --build $binDir --config Release --target qwen3_chatbot -- /m
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Built: $binDir"
