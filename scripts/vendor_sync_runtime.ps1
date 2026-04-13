param(
  [string]$ExecuTorchRoot = "C:/Apps/qwen3-export/third_party/executorch"
)

$ErrorActionPreference = 'Stop'
$repoRoot = Resolve-Path (Join-Path $PSScriptRoot '..')
$srcRoot = Join-Path (Resolve-Path $ExecuTorchRoot) 'examples\models\llama'
$dstRoot = Join-Path $repoRoot 'vendor-runtime'

$map = @(
  @{src='runner\runner.cpp'; dst='src\runner.cpp'},
  @{src='runner\runner.h'; dst='include\runner.h'},
  @{src='tokenizer\llama_tiktoken.cpp'; dst='src\llama_tiktoken.cpp'},
  @{src='tokenizer\llama_tiktoken.h'; dst='include\llama_tiktoken.h'}
)

foreach ($p in $map) {
  Copy-Item -Force (Join-Path $srcRoot $p.src) (Join-Path $dstRoot $p.dst)
  Copy-Item -Force (Join-Path $srcRoot $p.src) (Join-Path $dstRoot ('upstream\' + [IO.Path]::GetFileName($p.src)))
}

# Local include adjustments for vendored paths
(Get-Content (Join-Path $dstRoot 'src\runner.cpp')) -replace '<executorch/examples/models/llama/runner/runner.h>', '"runner.h"' |
  Set-Content (Join-Path $dstRoot 'src\runner.cpp') -Encoding UTF8
(Get-Content (Join-Path $dstRoot 'src\runner.cpp')) -replace '<executorch/examples/models/llama/tokenizer/llama_tiktoken.h>', '"llama_tiktoken.h"' |
  Set-Content (Join-Path $dstRoot 'src\runner.cpp') -Encoding UTF8
(Get-Content (Join-Path $dstRoot 'include\runner.h')) -replace '<executorch/examples/models/llama/tokenizer/llama_tiktoken.h>', '"llama_tiktoken.h"' |
  Set-Content (Join-Path $dstRoot 'include\runner.h') -Encoding UTF8
(Get-Content (Join-Path $dstRoot 'src\llama_tiktoken.cpp')) -replace '<executorch/examples/models/llama/tokenizer/llama_tiktoken.h>', '"llama_tiktoken.h"' |
  Set-Content (Join-Path $dstRoot 'src\llama_tiktoken.cpp') -Encoding UTF8

Write-Host "Vendor runtime sync complete: $dstRoot"
