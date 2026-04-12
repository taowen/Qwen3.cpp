[CmdletBinding()]
param(
    [string]$Model = "",
    [string]$Prompt = "Write one sentence about why cache locality matters in CPU performance.",
    [int]$NPredict = 64,
    [string]$Exe = "",
    [string]$BaselineFile = "",
    [string]$OutDir = "",
    [switch]$KeepLogs
)

$ErrorActionPreference = "Stop"
if (Get-Variable -Name PSNativeCommandUseErrorActionPreference -ErrorAction SilentlyContinue) {
    $PSNativeCommandUseErrorActionPreference = $false
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

if ([string]::IsNullOrWhiteSpace($Exe)) {
    $Exe = Join-Path $repoRoot "build-cli-fresh\Release\qwen3-cli.exe"
}
if ([string]::IsNullOrWhiteSpace($Model)) {
    $Model = Join-Path $repoRoot "models\Qwen3-0.6B-gguf\Qwen3-0.6B-752M-Q4_K_M.gguf"
}
if ([string]::IsNullOrWhiteSpace($BaselineFile)) {
    $BaselineFile = Join-Path $repoRoot "scripts\baselines\cpu-determinism.stdout.txt"
}
if ([string]::IsNullOrWhiteSpace($OutDir)) {
    $OutDir = Join-Path $repoRoot "runs\determinism"
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

New-Item -ItemType Directory -Path $OutDir -Force | Out-Null

function Invoke-Run {
    param(
        [Parameter(Mandatory = $true)][int]$Index,
        [Parameter(Mandatory = $true)][string]$ExePath,
        [Parameter(Mandatory = $true)][string]$ModelPath,
        [Parameter(Mandatory = $true)][string]$PromptText,
        [Parameter(Mandatory = $true)][int]$MaxTokens,
        [Parameter(Mandatory = $true)][string]$OutputDir
    )

    $stdoutPath = Join-Path $OutputDir ("stdout_run{0}.txt" -f $Index)
    $stderrPath = Join-Path $OutputDir ("stderr_run{0}.txt" -f $Index)

    $args = @(
        "-m", $ModelPath,
        "-n", $MaxTokens.ToString(),
        "-p", $PromptText
    )

    $proc = Start-Process -FilePath $ExePath -ArgumentList $args -NoNewWindow -Wait -PassThru `
        -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath

    return [PSCustomObject]@{
        Index = $Index
        ExitCode = $proc.ExitCode
        StdoutPath = $stdoutPath
        StderrPath = $stderrPath
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

Write-Host "Running determinism test..."
Write-Host "Exe: $Exe"
Write-Host "Model: $Model"
Write-Host "NPredict: $NPredict"
Write-Host "BaselineFile: $BaselineFile"
Write-Host "OutDir: $OutDir"
Write-Host "Prompt: $Prompt"

$run1 = Invoke-Run -Index 1 -ExePath $Exe -ModelPath $Model -PromptText $Prompt -MaxTokens $NPredict -OutputDir $OutDir
$run2 = Invoke-Run -Index 2 -ExePath $Exe -ModelPath $Model -PromptText $Prompt -MaxTokens $NPredict -OutputDir $OutDir

if ($run1.ExitCode -ne 0 -or $run2.ExitCode -ne 0) {
    Write-Host "FAIL: CLI returned non-zero exit code."
    Write-Host ("run1 exit={0}, run2 exit={1}" -f $run1.ExitCode, $run2.ExitCode)
    Write-Host ("stderr logs: {0}, {1}" -f $run1.StderrPath, $run2.StderrPath)
    exit 1
}

$run1Text = Get-NormalizedText -Path $run1.StdoutPath
$run2Text = Get-NormalizedText -Path $run2.StdoutPath
$baselineText = Get-NormalizedText -Path $BaselineFile

if ($run1Text -cne $run2Text) {
    Write-Host "FAIL: stdout differs between two runs."
    Write-Host ("stdout logs: {0}, {1}" -f $run1.StdoutPath, $run2.StdoutPath)
    $diff = Get-FirstDiff -Left $run1Text -Right $run2Text
    if ($diff) {
        Write-Host ("First diff line: {0}" -f $diff.Line)
        Write-Host ("run1: {0}" -f $diff.Left)
        Write-Host ("run2: {0}" -f $diff.Right)
    }
    Write-Host "Note: stderr is not compared because performance timing is expected to vary."
    exit 2
}

if ($run1Text -cne $baselineText) {
    Write-Host "FAIL: stdout text does not match pinned baseline text."
    Write-Host ("stdout logs: {0}, {1}" -f $run1.StdoutPath, $run2.StdoutPath)
    Write-Host ("baseline file: {0}" -f $BaselineFile)
    $baselineDiff = Get-FirstDiff -Left $baselineText -Right $run1Text
    if ($baselineDiff) {
        Write-Host ("First diff line: {0}" -f $baselineDiff.Line)
        Write-Host ("baseline: {0}" -f $baselineDiff.Left)
        Write-Host ("actual:   {0}" -f $baselineDiff.Right)
    }
    Write-Host "Note: if model/weights/prompt/runtime changed intentionally, update the pinned baseline text file."
    exit 3
}

Write-Host "PASS: deterministic stdout across two runs and matches pinned baseline text."
Write-Host "Note: stderr is not compared because performance timing is expected to vary."

if (-not $KeepLogs) {
    Remove-Item -LiteralPath $run1.StdoutPath, $run1.StderrPath, $run2.StdoutPath, $run2.StderrPath -Force -ErrorAction SilentlyContinue
}

exit 0
