[CmdletBinding()]
param(
    [ValidateSet("xpu", "cpu")]
    [string]$Backend = "xpu",
    [string]$PythonVersion = "3.11",
    [string]$VenvPath = "",
    [switch]$Recreate
)

$ErrorActionPreference = "Stop"
if (Get-Variable -Name PSNativeCommandUseErrorActionPreference -ErrorAction SilentlyContinue) {
    $PSNativeCommandUseErrorActionPreference = $false
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if ([string]::IsNullOrWhiteSpace($VenvPath)) {
    $VenvPath = Join-Path $repoRoot ".venv"
}
$pythonExe = Join-Path $VenvPath "Scripts\python.exe"

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "uv not found in PATH"
}

if ($Recreate -and (Test-Path -LiteralPath $VenvPath)) {
    Remove-Item -LiteralPath $VenvPath -Recurse -Force
}

if (-not (Test-Path -LiteralPath $pythonExe)) {
    Write-Host "Creating uv venv: $VenvPath"
    & uv venv $VenvPath --python $PythonVersion
    if ($LASTEXITCODE -ne 0) {
        throw "uv venv failed"
    }
}

Write-Host "Installing Python dependencies from pyproject.toml ..."
& uv sync --project $repoRoot --python $pythonExe
if ($LASTEXITCODE -ne 0) {
    throw "uv sync failed"
}

if ($Backend -eq "xpu") {
    Write-Host "Installing Intel XPU PyTorch ..."
    & uv pip install --python $pythonExe --index-url https://download.pytorch.org/whl/xpu torch==2.11.0+xpu torchvision==0.22.0+xpu torchaudio==2.11.0+xpu
    if ($LASTEXITCODE -ne 0) {
        throw "installing XPU torch failed"
    }
} else {
    Write-Host "Installing CPU PyTorch ..."
    & uv pip install --python $pythonExe --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio
    if ($LASTEXITCODE -ne 0) {
        throw "installing CPU torch failed"
    }
}

Write-Host "Done"
Write-Host "Python:"
Write-Host "  $pythonExe"
Write-Host "Test with:"
Write-Host "  $pythonExe -m auto_round --help"
