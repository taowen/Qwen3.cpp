Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Get-RepoRoot {
  param(
    [Parameter(Mandatory = $true)]
    [string]$FromPath
  )

  $candidate = [System.IO.Path]::GetFullPath((Join-Path $FromPath ".."))
  if (Test-Path (Join-Path $candidate "pyproject.toml")) {
    return $candidate
  }

  $candidate2 = [System.IO.Path]::GetFullPath((Join-Path $FromPath "..\.."))
  if (Test-Path (Join-Path $candidate2 "pyproject.toml")) {
    return $candidate2
  }

  throw "Unable to resolve repository root from: $FromPath"
}

function Resolve-FullPath {
  param(
    [Parameter(Mandatory = $true)]
    [string]$Path,
    [Parameter(Mandatory = $true)]
    [string]$BaseDir
  )

  if ([string]::IsNullOrWhiteSpace($Path)) {
    return $Path
  }

  if ([System.IO.Path]::IsPathRooted($Path)) {
    return [System.IO.Path]::GetFullPath($Path)
  }

  return [System.IO.Path]::GetFullPath((Join-Path $BaseDir $Path))
}

function Import-Qwen3Config {
  param(
    [Parameter(Mandatory = $true)]
    [string]$RepoRoot,
    [string]$ConfigPath = ""
  )

  $effectivePath = $ConfigPath
  if ([string]::IsNullOrWhiteSpace($effectivePath)) {
    $effectivePath = Join-Path $RepoRoot "config\default.psd1"
  } else {
    $effectivePath = Resolve-FullPath -Path $effectivePath -BaseDir $RepoRoot
  }

  if (!(Test-Path $effectivePath)) {
    throw "Config file not found: $effectivePath"
  }

  $cfg = Import-PowerShellDataFile -Path $effectivePath
  if ($null -eq $cfg) {
    throw "Failed to load config file: $effectivePath"
  }

  return @{
    Path = $effectivePath
    Data = $cfg
  }
}

function Ensure-Directory {
  param(
    [Parameter(Mandatory = $true)]
    [string]$Path
  )

  if (!(Test-Path $Path)) {
    New-Item -ItemType Directory -Force -Path $Path | Out-Null
  }
}

function Require-Path {
  param(
    [Parameter(Mandatory = $true)]
    [string]$Path,
    [string]$Description = "required path"
  )

  if (!(Test-Path $Path)) {
    throw "Missing ${Description}: $Path"
  }
}

function Invoke-Checked {
  param(
    [Parameter(Mandatory = $true)]
    [string]$Executable,
    [string[]]$Arguments = @(),
    [string]$WorkingDirectory = ""
  )

  $oldLocation = $null
  if (![string]::IsNullOrWhiteSpace($WorkingDirectory)) {
    $oldLocation = Get-Location
    Set-Location $WorkingDirectory
  }

  try {
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) {
      throw "Command failed (exit=$LASTEXITCODE): $Executable $($Arguments -join ' ')"
    }
  } finally {
    if ($null -ne $oldLocation) {
      Set-Location $oldLocation
    }
  }
}

function Invoke-RobocopyChecked {
  param(
    [Parameter(Mandatory = $true)]
    [string]$Source,
    [Parameter(Mandatory = $true)]
    [string]$Target,
    [string[]]$Options = @("/E", "/R:1", "/W:1")
  )

  & robocopy $Source $Target @Options
  $exitCode = $LASTEXITCODE
  if ($exitCode -ge 8) {
    throw "robocopy failed (exit=$exitCode): $Source -> $Target"
  }
}

function Write-Step {
  param(
    [Parameter(Mandatory = $true)]
    [string]$Message
  )

  Write-Host "[qwen3] $Message"
}
