param(
  [ValidateSet('pull', 'push')]
  [string]$Mode = 'pull',
  [string]$ExecuTorchRoot = "C:/Apps/qwen3-export/third_party/executorch"
)

$ErrorActionPreference = 'Stop'

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$upstreamRoot = Join-Path (Resolve-Path $ExecuTorchRoot) 'backends\vulkan'
$vendorRoot = Join-Path $repoRoot 'vendor-executorch\backends\vulkan'

if (!(Test-Path $upstreamRoot)) {
  throw "Upstream Vulkan backend path not found: $upstreamRoot"
}

if ($Mode -eq 'pull') {
  $src = $upstreamRoot
  $dst = $vendorRoot
} else {
  if (!(Test-Path $vendorRoot)) {
    throw "Vendor Vulkan backend path not found: $vendorRoot"
  }
  $src = $vendorRoot
  $dst = $upstreamRoot
}

New-Item -ItemType Directory -Force -Path $dst | Out-Null

robocopy $src $dst /E /R:1 /W:1 /XD __pycache__ /XF *.pyc
$exitCode = $LASTEXITCODE
if ($exitCode -ge 8) {
  throw "robocopy failed with exit code $exitCode"
}

Write-Host "Vulkan backend sync complete."
Write-Host "Mode: $Mode"
Write-Host "Source: $src"
Write-Host "Target: $dst"
