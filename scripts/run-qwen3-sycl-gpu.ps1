[CmdletBinding()]
param(
    [string]$Model = "C:\Apps\llama.cpp\models\Qwen3\Qwen3-0.6B-Q8_0.gguf",
    [string]$Prompt = "Write a concise explanation of why cache locality matters in CPU performance.",
    [int]$NPredict = 256,
    [int]$NGpuLayers = 999,
    [string]$Exe = "C:\Apps\Qwen3.cpp\build-sycl-oneapi-ninja-icx\qwen3-cli.exe",
    [string]$OneApiSetvars = "C:\Progra~2\Intel\oneAPI\setvars.bat",
    [string]$LogPath = ""
)

$ErrorActionPreference = "Stop"
if (Get-Variable -Name PSNativeCommandUseErrorActionPreference -ErrorAction SilentlyContinue) {
    $PSNativeCommandUseErrorActionPreference = $false
}

function Import-OneApiEnvironment {
    param([Parameter(Mandatory = $true)][string]$SetvarsPath)

    if (-not (Test-Path -LiteralPath $SetvarsPath)) {
        throw "oneAPI setvars not found: $SetvarsPath"
    }

    $envDump = & cmd.exe /d /s /c "`"$SetvarsPath`" >nul && set"
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to initialize oneAPI environment via: $SetvarsPath"
    }

    foreach ($line in $envDump) {
        $idx = $line.IndexOf("=")
        if ($idx -le 0) {
            continue
        }
        $name = $line.Substring(0, $idx)
        $value = $line.Substring($idx + 1)
        Set-Item -Path ("Env:" + $name) -Value $value
    }
}

if (-not (Test-Path -LiteralPath $Exe)) {
    throw "CLI executable not found: $Exe"
}
if (-not (Test-Path -LiteralPath $Model)) {
    throw "Model not found: $Model"
}
if ($NPredict -lt 1) {
    throw "NPredict must be >= 1"
}

Import-OneApiEnvironment -SetvarsPath $OneApiSetvars

$cliArgs = @(
    "-m", $Model,
    "-ngl", $NGpuLayers.ToString(),
    "-n", $NPredict.ToString(),
    "-p", $Prompt
)

Write-Host "Running SYCL GPU..."
Write-Host "Model: $Model"
Write-Host "NGpuLayers: $NGpuLayers"
Write-Host "NPredict: $NPredict"
if (-not [string]::IsNullOrWhiteSpace($LogPath)) {
    Write-Host "Log: $LogPath"
}

$stdoutPath = [System.IO.Path]::GetTempFileName()
$stderrPath = [System.IO.Path]::GetTempFileName()

try {
    $proc = Start-Process -FilePath $Exe -ArgumentList $cliArgs -NoNewWindow -Wait -PassThru `
        -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath

    $combined = @()
    if (Test-Path -LiteralPath $stdoutPath) {
        $combined += Get-Content -LiteralPath $stdoutPath
    }
    if (Test-Path -LiteralPath $stderrPath) {
        $combined += Get-Content -LiteralPath $stderrPath
    }

    if ([string]::IsNullOrWhiteSpace($LogPath)) {
        $combined | ForEach-Object { Write-Host $_ }
    } else {
        $combined | Tee-Object -FilePath $LogPath
    }

    $exitCode = $proc.ExitCode
}
finally {
    if (Test-Path -LiteralPath $stdoutPath) {
        Remove-Item -LiteralPath $stdoutPath -Force
    }
    if (Test-Path -LiteralPath $stderrPath) {
        Remove-Item -LiteralPath $stderrPath -Force
    }
}

Write-Host "ExitCode: $exitCode"
exit $exitCode
