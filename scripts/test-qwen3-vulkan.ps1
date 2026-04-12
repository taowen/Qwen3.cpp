[CmdletBinding()]
param(
    [string]$Model = "",
    [string]$Prompt = "Write one sentence about why cache locality matters.",
    [int]$NPredict = 64,
    [int]$NGpuLayers = 999,
    [string]$Exe = "",
    [string]$BaselineFile = "",
    [int]$TimeoutSec = 300,
    [string]$OutDir = "",
    [switch]$KeepLogs
)

$ErrorActionPreference = "Stop"
if (Get-Variable -Name PSNativeCommandUseErrorActionPreference -ErrorAction SilentlyContinue) {
    $PSNativeCommandUseErrorActionPreference = $false
}

function Invoke-Run {
    param(
        [Parameter(Mandatory = $true)][int]$Index,
        [Parameter(Mandatory = $true)][string]$ExePath,
        [Parameter(Mandatory = $true)][string[]]$Args,
        [Parameter(Mandatory = $true)][int]$TimeoutSeconds,
        [Parameter(Mandatory = $true)][string]$OutputDir
    )

    $stdoutPath = Join-Path $OutputDir ("vulkan_stdout_run{0}.txt" -f $Index)
    $stderrPath = Join-Path $OutputDir ("vulkan_stderr_run{0}.txt" -f $Index)

    $proc = Start-Process -FilePath $ExePath -ArgumentList $Args -NoNewWindow -PassThru `
        -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath

    $timedOut = $false
    if (-not $proc.WaitForExit($TimeoutSeconds * 1000)) {
        $timedOut = $true
        Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
        $null = $proc.WaitForExit(5000)
    }

    $exitCode = 0
    if (-not $timedOut) {
        $proc.Refresh()
        $exitCode = $proc.ExitCode
        if ($null -eq $exitCode) {
            $exitCode = 0
        }
    }

    return [PSCustomObject]@{
        Index = $Index
        TimedOut = $timedOut
        ExitCode = $exitCode
        StdoutPath = $stdoutPath
        StderrPath = $stderrPath
        StderrText = (Get-Content -LiteralPath $stderrPath -Raw)
        SpeedLine = (Get-Content -LiteralPath $stderrPath | Select-String -Pattern "^main: decoded .* speed: .* t/s$" | Select-Object -Last 1).Line
    }
}

function Get-NormalizedText {
    param([Parameter(Mandatory = $true)][string]$Path)

    $text = Get-Content -LiteralPath $Path -Raw
    $text = $text -replace "`r`n", "`n"
    $text = $text -replace "`r", "`n"
    return $text
}

function Get-FirstDiff {
    param(
        [Parameter(Mandatory = $true)][string]$Left,
        [Parameter(Mandatory = $true)][string]$Right
    )

    $leftLines = $Left -split "`n", -1
    $rightLines = $Right -split "`n", -1
    $maxLines = [Math]::Max($leftLines.Count, $rightLines.Count)

    for ($i = 0; $i -lt $maxLines; $i++) {
        $leftLine = if ($i -lt $leftLines.Count) { $leftLines[$i] } else { "<EOF>" }
        $rightLine = if ($i -lt $rightLines.Count) { $rightLines[$i] } else { "<EOF>" }
        if ($leftLine -cne $rightLine) {
            return [PSCustomObject]@{
                Line = $i + 1
                Left = $leftLine
                Right = $rightLine
            }
        }
    }

    return $null
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

if ([string]::IsNullOrWhiteSpace($Exe)) {
    $Exe = Join-Path $repoRoot "build-vulkan\Release\qwen3-cli.exe"
}
if ([string]::IsNullOrWhiteSpace($Model)) {
    $Model = Join-Path $repoRoot "models\Qwen3-0.6B-gguf\Qwen3-0.6B-752M-Q4_K_M.gguf"
}
if ([string]::IsNullOrWhiteSpace($BaselineFile)) {
    $BaselineFile = Join-Path $repoRoot "scripts\baselines\vulkan-gpu.stdout.txt"
}
if ([string]::IsNullOrWhiteSpace($OutDir)) {
    $OutDir = Join-Path $repoRoot "runs\vulkan-test"
}

if (-not (Test-Path -LiteralPath $Exe)) {
    throw "CLI executable not found: $Exe"
}
if (-not (Test-Path -LiteralPath $Model)) {
    throw "Model not found: $Model"
}
if (-not (Test-Path -LiteralPath $BaselineFile)) {
    throw "Baseline file not found: $BaselineFile"
}
if ($NPredict -lt 1) {
    throw "NPredict must be >= 1"
}
if ($NGpuLayers -lt 0) {
    throw "NGpuLayers must be >= 0"
}
if ($TimeoutSec -lt 1) {
    throw "TimeoutSec must be >= 1"
}

New-Item -ItemType Directory -Path $OutDir -Force | Out-Null

$args = @(
    "-m", $Model,
    "-ngl", $NGpuLayers.ToString(),
    "-n", $NPredict.ToString(),
    "-p", $Prompt
)

Write-Host "Running Vulkan GPU determinism test..."
Write-Host "Exe: $Exe"
Write-Host "Model: $Model"
Write-Host "NGpuLayers: $NGpuLayers"
Write-Host "NPredict: $NPredict"
Write-Host "BaselineFile: $BaselineFile"
Write-Host "OutDir: $OutDir"

$run1 = Invoke-Run -Index 1 -ExePath $Exe -Args $args -TimeoutSeconds $TimeoutSec -OutputDir $OutDir
$run2 = Invoke-Run -Index 2 -ExePath $Exe -Args $args -TimeoutSeconds $TimeoutSec -OutputDir $OutDir

if ($run1.TimedOut -or $run2.TimedOut) {
    Write-Host "FAIL: at least one run timed out."
    Write-Host ("run1 timedOut={0}, run2 timedOut={1}" -f $run1.TimedOut, $run2.TimedOut)
    Write-Host ("stderr logs: {0}, {1}" -f $run1.StderrPath, $run2.StderrPath)
    exit 124
}

if ($run1.ExitCode -ne 0 -or $run2.ExitCode -ne 0) {
    Write-Host "FAIL: non-zero exit code."
    Write-Host ("run1 exit={0}, run2 exit={1}" -f $run1.ExitCode, $run2.ExitCode)
    Write-Host ("stderr logs: {0}, {1}" -f $run1.StderrPath, $run2.StderrPath)
    exit 1
}

$checks = @(
    @{ Name = "Vulkan device discovery"; Pattern = "ggml_vulkan:\s+Found\s+\d+\s+Vulkan devices:" },
    @{ Name = "Layer offload to GPU"; Pattern = "load_tensors:\s+offloaded\s+\d+/\d+\s+layers to GPU" },
    @{ Name = "Vulkan KV cache device"; Pattern = "dev\s*=\s*Vulkan\d+" }
)

foreach ($run in @($run1, $run2)) {
    foreach ($c in $checks) {
        if (-not ([regex]::IsMatch($run.StderrText, $c.Pattern))) {
            Write-Host ("FAIL: run{0} missing expected log pattern: {1}" -f $run.Index, $c.Name)
            Write-Host ("Pattern: {0}" -f $c.Pattern)
            Write-Host ("stderr log: {0}" -f $run.StderrPath)
            exit 2
        }
    }
}

$run1Text = Get-NormalizedText -Path $run1.StdoutPath
$run2Text = Get-NormalizedText -Path $run2.StdoutPath
$baselineText = Get-NormalizedText -Path $BaselineFile

if ($run1Text -cne $run2Text) {
    Write-Host "FAIL: Vulkan stdout differs across two identical runs."
    Write-Host ("stdout logs: {0}, {1}" -f $run1.StdoutPath, $run2.StdoutPath)
    $diff = Get-FirstDiff -Left $run1Text -Right $run2Text
    if ($diff) {
        Write-Host ("First diff line: {0}" -f $diff.Line)
        Write-Host ("run1: {0}" -f $diff.Left)
        Write-Host ("run2: {0}" -f $diff.Right)
    }
    Write-Host "Note: stderr is not compared because timing/perf logs vary by run."
    exit 3
}

if ($run1Text -cne $baselineText) {
    Write-Host "FAIL: Vulkan stdout text does not match pinned baseline text."
    Write-Host ("stdout logs: {0}, {1}" -f $run1.StdoutPath, $run2.StdoutPath)
    Write-Host ("baseline file: {0}" -f $BaselineFile)
    $baselineDiff = Get-FirstDiff -Left $baselineText -Right $run1Text
    if ($baselineDiff) {
        Write-Host ("First diff line: {0}" -f $baselineDiff.Line)
        Write-Host ("baseline: {0}" -f $baselineDiff.Left)
        Write-Host ("actual:   {0}" -f $baselineDiff.Right)
    }
    Write-Host "Note: if model/weights/prompt/runtime changed intentionally, update the pinned baseline text file."
    exit 4
}

Write-Host "PASS: Vulkan backend is active, deterministic, and matches pinned baseline text."
if ($run1.SpeedLine) { Write-Host ("run1: {0}" -f $run1.SpeedLine) }
if ($run2.SpeedLine) { Write-Host ("run2: {0}" -f $run2.SpeedLine) }

if (-not $KeepLogs) {
    Remove-Item -LiteralPath $run1.StdoutPath, $run1.StderrPath, $run2.StdoutPath, $run2.StderrPath -Force -ErrorAction SilentlyContinue
}

exit 0
