param(
    [string[]]$Roots = @("src", "ggml/src"),
    [string[]]$Include = @("*.c", "*.cc", "*.cpp", "*.cxx"),
    [switch]$Json
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Remove-Comments {
    param([string]$Text)

    $noBlock = [regex]::Replace(
        $Text,
        '/\*.*?\*/',
        '',
        [System.Text.RegularExpressions.RegexOptions]::Singleline
    )
    return [regex]::Replace($noBlock, '//.*', '')
}

function Get-StaticFunctionNames {
    param([string]$Text)

    $pattern = '(?m)^\s*static\s+(?:inline\s+)?(?:[\w:<>*&\[\]\s]+?)\s+([A-Za-z_]\w*)\s*\('
    return [regex]::Matches($Text, $pattern) |
        ForEach-Object { $_.Groups[1].Value } |
        Sort-Object -Unique
}

$repoRoot = (Get-Location).Path
$files = New-Object System.Collections.Generic.List[System.IO.FileInfo]

foreach ($root in $Roots) {
    $abs = if ([System.IO.Path]::IsPathRooted($root)) { $root } else { Join-Path $repoRoot $root }
    if (-not (Test-Path -LiteralPath $abs)) {
        continue
    }
    Get-ChildItem -Path $abs -Recurse -File -Include $Include | ForEach-Object {
        $files.Add($_) | Out-Null
    }
}

$results = New-Object System.Collections.Generic.List[object]
foreach ($file in $files) {
    $raw = Get-Content -Raw -LiteralPath $file.FullName
    $src = Remove-Comments -Text $raw
    $defs = Get-StaticFunctionNames -Text $src
    foreach ($name in $defs) {
        $escaped = [regex]::Escape($name)
        $refCount = ([regex]::Matches($src, "\b$escaped\b")).Count
        if ($refCount -le 1) {
            $results.Add([PSCustomObject]@{
                file = $file.FullName
                name = $name
                refs = $refCount
            })
        }
    }
}

$ordered = $results | Sort-Object file, name
if ($Json) {
    $ordered | ConvertTo-Json -Depth 3
} else {
    $ordered | Format-Table -AutoSize
}
