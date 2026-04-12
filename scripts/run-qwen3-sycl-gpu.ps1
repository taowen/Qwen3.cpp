[CmdletBinding()]
param(
    [string]$Model = "C:\Apps\llama.cpp\models\Qwen3\Qwen3-0.6B-Q8_0.gguf",
    [string]$Prompt = "Write a concise explanation of why cache locality matters in CPU performance.",
    [int]$NPredict = 256,
    [int]$NGpuLayers = 999,
    [string]$Exe = "C:\Apps\Qwen3.cpp\build-sycl-oneapi-ninja-icx\qwen3-cli.exe",
    [string]$OneApiSetvars = "C:\Progra~2\Intel\oneAPI\setvars.bat",
    [ValidateSet("on", "off")]
    [string]$Graph = "on",
    [int]$TimeoutSec = 300,
    [int]$TailLines = 120,
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
if ($Exe -match "build-sycl-oneapi-vs-intel") {
    throw "Unsupported SYCL executable path ($Exe). Use Ninja build output (build-sycl-oneapi-ninja-icx\\qwen3-cli.exe)."
}
if (-not (Test-Path -LiteralPath $Model)) {
    throw "Model not found: $Model"
}
if ($NPredict -lt 1) {
    throw "NPredict must be >= 1"
}
if ($TimeoutSec -lt 1) {
    throw "TimeoutSec must be >= 1"
}
if ($TailLines -lt 1) {
    throw "TailLines must be >= 1"
}

Import-OneApiEnvironment -SetvarsPath $OneApiSetvars

if ($Graph -eq "on") {
    $env:GGML_SYCL_DISABLE_GRAPH = "0"
} else {
    $env:GGML_SYCL_DISABLE_GRAPH = "1"
}

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
Write-Host ("GGML_SYCL_DISABLE_GRAPH: {0}" -f $env:GGML_SYCL_DISABLE_GRAPH)
Write-Host "TimeoutSec: $TimeoutSec"
if (-not [string]::IsNullOrWhiteSpace($LogPath)) {
    Write-Host "Log: $LogPath"
}

$stdoutPath = [System.IO.Path]::GetTempFileName()
$stderrPath = [System.IO.Path]::GetTempFileName()

try {
    $proc = Start-Process -FilePath $Exe -ArgumentList $cliArgs -NoNewWindow -PassThru `
        -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath

    $timedOut = $false
    if (-not $proc.WaitForExit($TimeoutSec * 1000)) {
        $timedOut = $true
        Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
        $null = $proc.WaitForExit(5000)
    }

    if ([string]::IsNullOrWhiteSpace($LogPath)) {
        $outTail = @()
        $errTail = @()
        if (Test-Path -LiteralPath $stdoutPath) {
            $outTail = @(Get-Content -LiteralPath $stdoutPath -Encoding UTF8 -Tail $TailLines)
        }
        if (Test-Path -LiteralPath $stderrPath) {
            $errTail = @(Get-Content -LiteralPath $stderrPath -Encoding UTF8 -Tail $TailLines)
        }
        if ($outTail.Count -gt 0) {
            Write-Host "---- stdout tail ($TailLines) ----"
            $outTail | ForEach-Object { Write-Host $_ }
        }
        if ($errTail.Count -gt 0) {
            Write-Host "---- stderr tail ($TailLines) ----"
            $errTail | ForEach-Object { Write-Host $_ }
        }
    } else {
        $combined = @()
        if (Test-Path -LiteralPath $stdoutPath) {
            $combined += Get-Content -LiteralPath $stdoutPath -Encoding UTF8
        }
        if (Test-Path -LiteralPath $stderrPath) {
            $combined += Get-Content -LiteralPath $stderrPath -Encoding UTF8
        }
        $combined | Set-Content -LiteralPath $LogPath -Encoding UTF8
        Write-Host "Log written to: $LogPath"
        $tail = @(Get-Content -LiteralPath $LogPath -Encoding UTF8 -Tail $TailLines)
        if ($tail.Count -gt 0) {
            Write-Host "---- log tail ($TailLines) ----"
            $tail | ForEach-Object { Write-Host $_ }
        }
    }

    if ($timedOut) {
        $exitCode = 124
        Write-Host "Process timed out after $TimeoutSec seconds and was terminated."
    } else {
        $proc.Refresh()
        $exitCode = $proc.ExitCode
        if ($null -eq $exitCode) {
            $exitCode = 0
        }
    }
}
finally {
    foreach ($tmpPath in @($stdoutPath, $stderrPath)) {
        if (-not (Test-Path -LiteralPath $tmpPath)) {
            continue
        }
        $removed = $false
        for ($i = 0; $i -lt 5 -and -not $removed; $i++) {
            try {
                Remove-Item -LiteralPath $tmpPath -Force -ErrorAction Stop
                $removed = $true
            } catch {
                Start-Sleep -Milliseconds 100
            }
        }
    }
}

Write-Host "ExitCode: $exitCode"
exit $exitCode
