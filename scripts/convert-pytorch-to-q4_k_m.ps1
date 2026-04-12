[CmdletBinding()]
param(
    [string]$Model = "Qwen/Qwen3-0.6B",
    [string]$OutDir = "",
    [string]$PythonExe = "",
    [string]$DeviceMap = "xpu",
    [int]$Iters = 0,
    [switch]$SkipPythonSetup
)

$ErrorActionPreference = "Stop"
if (Get-Variable -Name PSNativeCommandUseErrorActionPreference -ErrorAction SilentlyContinue) {
    $PSNativeCommandUseErrorActionPreference = $false
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if ([string]::IsNullOrWhiteSpace($OutDir)) {
    $OutDir = Join-Path $repoRoot "models\qwen3-0.6b-q4_k_m"
}
if ($Iters -lt 0) {
    throw "Iters must be >= 0"
}

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

function Resolve-PythonExe {
    param(
        [string]$Requested,
        [string]$RepoRoot,
        [string]$DeviceMap,
        [switch]$DisableSetup
    )

    if (-not [string]::IsNullOrWhiteSpace($Requested)) {
        if (-not (Test-Path -LiteralPath $Requested)) {
            throw "Python executable not found: $Requested"
        }
        return $Requested
    }

    $localVenv = Join-Path $RepoRoot ".venv"
    $localPy = Join-Path $localVenv "Scripts\python.exe"
    if (Test-Path -LiteralPath $localPy) {
        return $localPy
    }

    $preferred = "C:\Apps\auto-round\.venv-xpu\Scripts\python.exe"
    if ($DeviceMap -match "^xpu" -and (Test-Path -LiteralPath $preferred)) {
        return $preferred
    }

    if ($DisableSetup) {
        throw "No Python environment found for AutoRound. Provide -PythonExe or remove -SkipPythonSetup."
    }

    $setupScript = Join-Path $RepoRoot "scripts\setup-uv-env.ps1"
    if (-not (Test-Path -LiteralPath $setupScript)) {
        throw "setup script not found: $setupScript"
    }

    $backend = "cpu"
    if ($DeviceMap -match "^xpu") {
        $backend = "xpu"
    }

    & powershell -NoProfile -ExecutionPolicy Bypass -File $setupScript -Backend $backend -VenvPath $localVenv
    if ($LASTEXITCODE -ne 0) {
        throw "setup-uv-env.ps1 failed"
    }

    if (-not (Test-Path -LiteralPath $localPy)) {
        throw "setup completed but python executable missing: $localPy"
    }

    return $localPy
}

$py = Resolve-PythonExe -Requested $PythonExe -RepoRoot $repoRoot -DeviceMap $DeviceMap -DisableSetup:$SkipPythonSetup

Write-Host "Running AutoRound -> GGUF:Q4_K_M ..."
Write-Host "Model:     $Model"
Write-Host "DeviceMap: $DeviceMap"
Write-Host "Iters:     $Iters"
Write-Host "OutDir:    $OutDir"

& $py -m auto_round `
    --model_name $Model `
    --format gguf:q4_k_m `
    --output_dir $OutDir `
    --device_map $DeviceMap `
    --iters $Iters
if ($LASTEXITCODE -ne 0) {
    throw "AutoRound quantization failed"
}

$gguf = Get-ChildItem -Path $OutDir -Recurse -Filter *.gguf -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1
if (-not $gguf) {
    throw "AutoRound completed but no .gguf file was found under: $OutDir"
}

Write-Host "Done"
Write-Host "Output:"
Write-Host "  $($gguf.FullName)"
